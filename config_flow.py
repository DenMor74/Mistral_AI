"""Config flow for Mistral AI conversation integration."""
import logging
import voluptuous as vol
from mistralai import Mistral
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import config_validation as cv

from .const import (
    DOMAIN,
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

class MistralConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Mistral AI conversation."""

    VERSION = 1

    async def async_step_user(self, user_input: dict | None = None) -> FlowResult:
        """Handle the initial step."""
        errors = {}
        if user_input is not None:
            try:
                # Offload Mistral client initialization and model listing to executor
                await self.hass.async_add_executor_job(
                    self._validate_api_key, user_input[CONF_API_KEY]
                )
            except Exception as err:
                _LOGGER.error("Failed to validate API key: %s", err)
                errors["base"] = "invalid_api_key"
            else:
                _LOGGER.debug("Creating config entry with input: %s", user_input)
                return self.async_create_entry(
                    title="Mistral AI Conversation",
                    data={
                        CONF_API_KEY: user_input[CONF_API_KEY],
                        CONF_MODEL: user_input[CONF_MODEL],
                        CONF_TEMPERATURE: user_input[CONF_TEMPERATURE],
                        CONF_TOP_P: user_input[CONF_TOP_P],
                        CONF_MAX_TOKENS: user_input[CONF_MAX_TOKENS],
                        CONF_CHAT_HISTORY: user_input[CONF_CHAT_HISTORY],
                        CONF_PROMPT: user_input[CONF_PROMPT],
                    },
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): cv.string,
                    vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): cv.string,
                    vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.Coerce(float),
                    vol.Optional(CONF_TOP_P, default=DEFAULT_TOP_P): vol.Coerce(float),
                    vol.Optional(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): vol.Coerce(int),
                    vol.Optional(CONF_CHAT_HISTORY, default=DEFAULT_CHAT_HISTORY): vol.Coerce(int),
                    vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
                }
            ),
            errors=errors,
        )

    def _validate_api_key(self, api_key: str) -> None:
        """Validate the API key by initializing Mistral client and listing models."""
        client = Mistral(api_key=api_key)
        client.models.list()

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return MistralOptionsFlow(config_entry)

class MistralOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Mistral AI conversation."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict | None = None) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            _LOGGER.debug("Saving options: %s", user_input)
            return self.async_create_entry(title="", data=user_input)

        # Use config_entry.options if available, otherwise fall back to config_entry.data
        current_options = self.config_entry.options if self.config_entry.options else self.config_entry.data
        _LOGGER.debug("Current options: %s", current_options)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MODEL,
                        default=current_options.get(CONF_MODEL, DEFAULT_MODEL),
                    ): cv.string,
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=current_options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                    ): vol.Coerce(float),
                    vol.Optional(
                        CONF_TOP_P,
                        default=current_options.get(CONF_TOP_P, DEFAULT_TOP_P),
                    ): vol.Coerce(float),
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=current_options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                    ): vol.Coerce(int),
                    vol.Optional(
                        CONF_CHAT_HISTORY,
                        default=current_options.get(CONF_CHAT_HISTORY, DEFAULT_CHAT_HISTORY),
                    ): vol.Coerce(int),
                    vol.Optional(
                        CONF_PROMPT,
                        default=current_options.get(CONF_PROMPT, DEFAULT_PROMPT),
                    ): cv.string,
                }
            ),
        )