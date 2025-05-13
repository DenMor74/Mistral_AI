"""Mistral AI conversation platform for Home Assistant."""
import logging
import asyncio
from typing import Any, Dict

from mistralai import Mistral
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.util import ulid
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    CONF_API_KEY,
    CONF_MODEL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_TOKENS,
    CONF_CHAT_HISTORY,
    CONF_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_CHAT_HISTORY,
    DEFAULT_PROMPT,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Set up the Mistral AI conversation entity."""
    try:
        mistral_client = await hass.async_add_executor_job(
            create_mistral_client, entry.data[CONF_API_KEY]
        )
        # Test API key by listing models
        await hass.async_add_executor_job(mistral_client.models.list)
    except Exception as err:
        _LOGGER.error("Failed to initialize Mistral AI client: %s", err)
        raise ConfigEntryNotReady(f"Failed to initialize Mistral AI client: {err}") from err

    entity = MistralConversationEntity(hass, entry, mistral_client)
    async_add_entities([entity])
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entity
    _LOGGER.debug("Successfully registered Mistral AI conversation entity with ID: %s", entity.entity_id)

def create_mistral_client(api_key: str) -> Mistral:
    """Create a Mistral client with the given API key."""
    try:
        return Mistral(api_key=api_key)
    except Exception as err:
        _LOGGER.error("Failed to create Mistral client: %s", err)
        raise

class ConversationResponse:
    """Temporary class to mimic ConversationResponse with as_dict method."""
    def __init__(self, speech: Dict[str, Any]) -> None:
        self.speech = speech

    def as_dict(self) -> Dict[str, Any]:
        """Return the response as a dictionary."""
        return {"speech": self.speech}

class MistralConversationEntity(conversation.ConversationEntity):
    """Mistral AI conversation entity."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, client: Mistral) -> None:
        """Initialize the entity."""
        self.hass = hass
        self.entry = entry
        self._client = client
        self.history: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}"
        self._attr_name = "Mistral AI Conversation"
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._is_ready = False

    async def async_added_to_hass(self) -> None:
        """Run when entity is added to Home Assistant."""
        try:
            # Verify client is functional
            await self.hass.async_add_executor_job(self._client.models.list)
            self._is_ready = True
            _LOGGER.debug("Mistral AI client initialized successfully for entity: %s", self.entity_id)
        except Exception as err:
            self._is_ready = False
            _LOGGER.error("Failed to initialize Mistral client for entity %s: %s", self.entity_id, err)
            raise ConfigEntryNotReady(f"Failed to initialize Mistral client: %s", err) from err

    @property
    def is_ready(self) -> bool:
        """Return if the entity is ready."""
        _LOGGER.debug("Checking is_ready for entity %s: client=%s, is_ready=%s", 
                     self.entity_id, self._client is not None, self._is_ready)
        return self._client is not None and self._is_ready

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return ["en", "es", "fr", "de", "it", "ru", "zh", "ja", "ko"]

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        """Process a user input and return the response."""
        _LOGGER.debug("Processing conversation input for entity %s: text=%s, conversation_id=%s", 
                     self.entity_id, user_input.text, user_input.conversation_id)
        if self._client is None or not self._is_ready:
            try:
                await self.async_added_to_hass()
            except ConfigEntryNotReady as err:
                _LOGGER.error("Failed to initialize client during process for entity %s: %s", self.entity_id, err)
                raise HomeAssistantError(f"Cannot process conversation: {err}") from err

        conversation_id = user_input.conversation_id or ulid.ulid()
        _LOGGER.debug("Using conversation_id=%s for entity %s", conversation_id, self.entity_id)

        # Use config_entry.options if available, otherwise fall back to config_entry.data
        settings = self.entry.options if self.entry.options else self.entry.data
        settings = {
            "model": settings.get(CONF_MODEL, DEFAULT_MODEL),
            "temperature": settings.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            "top_p": settings.get(CONF_TOP_P, DEFAULT_TOP_P),
            "max_tokens": settings.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            "chat_history": settings.get(CONF_CHAT_HISTORY, DEFAULT_CHAT_HISTORY),
            "prompt": settings.get(CONF_PROMPT, DEFAULT_PROMPT),
        }
        _LOGGER.debug("Using settings for entity %s: %s", self.entity_id, settings)

        async with self._lock:
            if conversation_id not in self.history:
                self.history[conversation_id] = []
                _LOGGER.debug("Initialized new history for conversation_id=%s", conversation_id)
            else:
                _LOGGER.debug("Existing history for conversation_id=%s: %s", 
                             conversation_id, self.history[conversation_id])

            # Include system prompt if defined
            messages = []
            if settings["prompt"]:
                messages.append({"role": "system", "content": settings["prompt"]})
            messages.extend(self.history[conversation_id][-settings["chat_history"]:])
            messages.append({"role": "user", "content": user_input.text})
            _LOGGER.debug("Messages sent to Mistral API for entity %s: %s", self.entity_id, messages)

            try:
                response = await self._client.chat.complete_async(
                    model=settings["model"],
                    messages=messages,
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    max_tokens=settings["max_tokens"],
                )

                if not response.choices:
                    _LOGGER.error("No response choices from Mistral AI API for entity %s", self.entity_id)
                    raise HomeAssistantError("No response from Mistral AI API")

                response_text = response.choices[0].message.content
                if not isinstance(response_text, str):
                    _LOGGER.error("Invalid response text type for entity %s: %s", self.entity_id, type(response_text))
                    raise HomeAssistantError("Invalid response text from Mistral AI API")

                _LOGGER.debug("Mistral AI response received for entity %s: %s", self.entity_id, response_text)

                # Update history
                self.history[conversation_id].extend([
                    {"role": "user", "content": user_input.text},
                    {"role": "assistant", "content": response_text},
                ])

                # Trim history to avoid excessive memory usage
                max_history = settings["chat_history"] * 2
                if len(self.history[conversation_id]) > max_history:
                    self.history[conversation_id] = self.history[conversation_id][-max_history:]
                    _LOGGER.debug("Trimmed history for conversation_id=%s to %s messages", 
                                 conversation_id, max_history)

                _LOGGER.debug("Updated history for conversation_id=%s: %s", 
                             conversation_id, self.history[conversation_id])

                return conversation.ConversationResult(
                    response=ConversationResponse(
                        speech={"plain": {"speech": response_text}}
                    ),
                    conversation_id=conversation_id,
                )

            except Exception as err:
                _LOGGER.error("Error processing Mistral AI request for entity %s: %s", self.entity_id, err)
                raise HomeAssistantError(f"Error processing request: {err}") from err