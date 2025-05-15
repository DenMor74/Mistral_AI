"""Платформа разговоров Mistral AI для Home Assistant."""
import logging
import asyncio
import json
from typing import Any

_LOGGER = logging.getLogger(__name__)

from mistralai import Mistral
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.util import ulid
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.const import CONF_LLM_HASS_API

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

try:
    from homeassistant.helpers import intent, llm
    LLM_AVAILABLE = True
    _LOGGER.debug("Интеграция LLM доступна")
except ImportError:
    LLM_AVAILABLE = False
    _LOGGER.warning("Интеграция LLM недоступна; поддержка llm_hass_api будет отключена")

# Максимальное количество итераций для вызовов инструментов
MAX_TOOL_ITERATIONS = 10

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Настройка сущности разговоров Mistral AI."""
    _LOGGER.debug("Попытка настройки сущности разговоров Mistral AI")
    client = entry.runtime_data
    try:
        models = await hass.async_add_executor_job(client.models.list)
        model_ids = [model.id for model in models.data]
        selected_model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
        if selected_model not in model_ids:
            _LOGGER.error("Модель %s недоступна. Доступные модели: %s", selected_model, model_ids)
            raise ConfigEntryNotReady(f"Модель {selected_model} недоступна")
        _LOGGER.debug("Доступные модели Mistral: %s", model_ids)
    except Exception as err:
        _LOGGER.error("Ошибка инициализации клиента Mistral AI: %s", err)
        raise ConfigEntryNotReady(f"Ошибка инициализации клиента Mistral AI: {err}") from err

    entity = MistralConversationEntity(hass, entry, client)
    async_add_entities([entity])
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entity
    _LOGGER.debug("Успешная регистрация сущности разговоров Mistral AI с ID: %s", entity.entity_id)

class ConversationResponse:
    """Временный класс для имитации ConversationResponse с методом as_dict."""
    def __init__(self, speech: dict[str, Any]) -> None:
        self.speech = speech
    def as_dict(self) -> dict[str, Any]:
        return {"speech": self.speech}

SUPPORTED_SCHEMA_KEYS = {
    "type",
    "description",
    "enum",
    "properties",
    "required",
    "items",
}

def _format_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Форматирование схемы для совместимости с Mistral API."""
    result = {}
    for key, val in schema.items():
        if key not in SUPPORTED_SCHEMA_KEYS:
            continue
        if key == "type":
            val = val.upper()
        elif key == "items":
            val = _format_schema(val)
        elif key == "properties":
            val = {k: _format_schema(v) for k, v in val.items()}
        result[key] = val
    if result.get("enum") and result.get("type") != "STRING":
        result["type"] = "STRING"
        result["enum"] = [str(item) for item in result["enum"]]
    return result

def _format_tool(tool: llm.Tool) -> dict[str, Any]:
    """Форматирование спецификации инструмента."""
    parameters = _format_schema(tool.parameters.schema) if tool.parameters.schema else {}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }

class MistralConversationEntity(conversation.ConversationEntity):
    """Сущность разговоров Mistral AI."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, client: Mistral) -> None:
        """Инициализация сущности."""
        self.hass = hass
        self.entry = entry
        self._client = client
        self.history: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}"
        self._attr_entity_id = "conversation.mistral_ai_conversation"
        self._attr_name = "Mistral AI Conversation"
        self._attr_has_entity_name = True
        self._attr_should_poll = False
        self._is_ready = False
        self._attr_supported_features = (
            conversation.ConversationEntityFeature.CONTROL
            if LLM_AVAILABLE and entry.options.get(CONF_LLM_HASS_API)
            else 0
        )
        _LOGGER.debug("Инициализирована MistralConversationEntity с entity_id: %s, unique_id: %s, llm_available: %s", 
                      self._attr_entity_id, self._attr_unique_id, LLM_AVAILABLE)

    async def async_added_to_hass(self) -> None:
        """Выполняется при добавлении сущности в Home Assistant."""
        _LOGGER.debug("Добавление сущности %s в Home Assistant", self.entity_id)
        try:
            models = await self.hass.async_add_executor_job(self._client.models.list)
            model_ids = [model.id for model in models.data]
            selected_model = self.entry.options.get(CONF_MODEL, DEFAULT_MODEL)
            if selected_model not in model_ids:
                _LOGGER.error("Модель %s недоступна для сущности %s. Доступные модели: %s", 
                              selected_model, self.entity_id, model_ids)
                raise ConfigEntryNotReady(f"Модель {selected_model} недоступна")
            _LOGGER.debug("Клиент Mistral AI успешно инициализирован для сущности %s, доступные модели: %s", 
                          self.entity_id, model_ids)
            self._is_ready = True
            conversation.async_set_agent(self.hass, self.entry, self)
            _LOGGER.debug("Агент %s успешно зарегистрирован", self.entity_id)
        except Exception as err:
            if "authentication" in str(err).lower() or "api key" in str(err).lower():
                _LOGGER.error("Неверный ключ API Mistral для сущности %s: %s", self.entity_id, err)
                raise ConfigEntryAuthFailed("Неверный ключ API Mistral") from err
            self._is_ready = False
            _LOGGER.error("Ошибка инициализации клиента Mistral для сущности %s: %s", self.entity_id, err)
            raise ConfigEntryNotReady(f"Ошибка инициализации клиента Mistral: {err}") from err

    async def async_will_remove_from_hass(self) -> None:
        """Выполняется при удалении сущности из Home Assistant."""
        _LOGGER.debug("Удаление сущности %s из Home Assistant", self.entity_id)
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def is_ready(self) -> bool:
        """Проверка готовности сущности."""
        _LOGGER.debug("Проверка is_ready для сущности %s: client=%s, is_ready=%s", 
                      self.entity_id, self._client is not None, self._is_ready)
        return self._client is not None and self._is_ready

    @property
    def supported_languages(self) -> list[str]:
        """Список поддерживаемых языков."""
        return ["en", "es", "fr", "de", "it", "ru", "zh", "ja", "ko"]

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        """Обработка пользовательского ввода и возврат ответа."""
        _LOGGER.debug("Обработка ввода разговора для сущности %s: текст=%s, conversation_id=%s", 
                      self.entity_id, user_input.text, user_input.conversation_id)
        if self._client is None or not self._is_ready:
            try:
                await self.async_added_to_hass()
            except ConfigEntryNotReady as err:
                _LOGGER.error("Ошибка инициализации клиента во время обработки для сущности %s: %s", self.entity_id, err)
                raise HomeAssistantError(f"Невозможно обработать разговор: {err}") from err

        conversation_id = user_input.conversation_id or ulid.ulid()
        chat_log = None
        tools: list[dict[str, Any]] | None = None
        llm_api_ids = self.entry.options.get(CONF_LLM_HASS_API, []) if LLM_AVAILABLE else []
        if not isinstance(llm_api_ids, list):
            llm_api_ids = [llm_api_ids] if llm_api_ids else []

        if LLM_AVAILABLE and llm_api_ids:
            _LOGGER.debug("Попытка инициализации журнала чата LLM с llm_api_ids: %s", llm_api_ids)
            try:
                chat_log = conversation.ChatLog(self.hass, conversation_id, user_input.agent_id)
                for llm_api_id in llm_api_ids:
                    await chat_log.async_update_llm_data(
                        DOMAIN,
                        user_input,
                        llm_api_id,
                        self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
                    )
                    if chat_log.llm_api:
                        tools = [_format_tool(tool) for tool in chat_log.llm_api.tools]
                        if not tools:
                            _LOGGER.warning("Инструменты недоступны для LLM API ID: %s", llm_api_id)
                        _LOGGER.debug("Получено %d инструментов для LLM API %s: %s", len(tools), llm_api_id, 
                                      [tool["function"]["name"] for tool in tools] if tools else [])
                        break
                else:
                    _LOGGER.warning("Экземпляр LLM API недоступен для ID: %s", llm_api_ids)
                    chat_log = None
                    tools = None
            except Exception as err:
                _LOGGER.error("Ошибка инициализации журнала чата LLM или инструментов для llm_api_ids %s: %s", llm_api_ids, err)
                chat_log = None
                tools = None
        else:
            _LOGGER.debug("LLM недоступен или llm_api_ids не заданы (LLM_AVAILABLE=%s, llm_api_ids=%s); используется базовый разговор", 
                          LLM_AVAILABLE, llm_api_ids)

        settings = {
            "model": self.entry.options.get(CONF_MODEL, DEFAULT_MODEL),
            "temperature": self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            "top_p": self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
            "max_tokens": self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            "chat_history": self.entry.options.get(CONF_CHAT_HISTORY, DEFAULT_CHAT_HISTORY),
            "prompt": self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
        }
        _LOGGER.debug("Используемые настройки для сущности %s: %s", self.entity_id, settings)

        async with self._lock:
            if not chat_log:
                if conversation_id not in self.history:
                    self.history[conversation_id] = []
                    _LOGGER.debug("Инициализирована новая история для conversation_id=%s", conversation_id)

            messages = []
            if settings["prompt"]:
                messages.append({"role": "system", "content": settings["prompt"]})

            if chat_log:
                for content in chat_log.content[-settings["chat_history"]:]:
                    if content.content:
                        messages.append({"role": content.role, "content": content.content})
            else:
                messages.extend(self.history[conversation_id][-settings["chat_history"]:])

            messages.append({"role": "user", "content": user_input.text})
            _LOGGER.debug("Подготовлены сообщения для API Mistral: %s", messages)

            for iteration in range(MAX_TOOL_ITERATIONS):
                try:
                    response = await self._client.chat.complete_async(
                        model=settings["model"],
                        messages=messages,
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        max_tokens=settings["max_tokens"],
                        tools=tools,
                        tool_choice="auto" if tools else None,
                    )

                    if not response.choices:
                        _LOGGER.error("Нет вариантов ответа от API Mistral AI для сущности %s", self.entity_id)
                        raise HomeAssistantError("Нет ответа от API Mistral AI")

                    if response.usage:
                        chat_log.async_trace({
                            "stats": {
                                "input_tokens": response.usage.prompt_tokens,
                                "output_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens,
                            }
                        }) if chat_log else None
                        _LOGGER.debug("Статистика использования: %s", response.usage)

                except Exception as err:
                    _LOGGER.error("Ошибка API Mistral для сущности %s: %s", self.entity_id, err)
                    raise HomeAssistantError(f"Ошибка связи с Mistral AI: {err}") from err

                if LLM_AVAILABLE and tools and response.choices[0].message.tool_calls:
                    _LOGGER.debug("Обработка вызовов инструментов: %s", response.choices[0].message.tool_calls)
                    for tool_call in response.choices[0].message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as err:
                            _LOGGER.error("Неверные аргументы инструмента для %s: %s", tool_name, err)
                            raise HomeAssistantError(f"Неверные аргументы инструмента: {err}") from err

                        _LOGGER.debug("Вызов инструмента %s с аргументами: %s", tool_name, tool_args)
                        try:
                            tool_result = await chat_log.async_call_tool(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                platform=DOMAIN,
                                context=user_input.context,
                                user_prompt=user_input.text,
                                language=user_input.language,
                                assistant="conversation",
                                device_id=user_input.device_id,
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "content": json.dumps(tool_result),
                                    "tool_call_id": tool_call.id,
                                }
                            )
                            _LOGGER.debug("Результат инструмента для %s: %s", tool_name, tool_result)
                        except Exception as err:
                            _LOGGER.error("Ошибка выполнения инструмента %s: %s", tool_name, err)
                            raise HomeAssistantError(f"Ошибка выполнения инструмента: {err}") from err
                    continue

                final_content = response.choices[0].message.content
                if not isinstance(final_content, str):
                    _LOGGER.error("Неверный тип текста ответа для сущности %s: %s", self.entity_id, type(final_content))
                    raise HomeAssistantError("Неверный текст ответа от API Mistral AI")

                _LOGGER.debug("Финальный ответ Mistral AI: %s", final_content)
                break
            else:
                _LOGGER.error("Превышен лимит итераций вызовов инструментов для сущности %s", self.entity_id)
                raise HomeAssistantError("Превышен лимит итераций вызовов инструментов")

            if chat_log:
                try:
                    await chat_log.async_add_content(
                        role="assistant",
                        content=final_content,
                        agent_id=user_input.agent_id,
                    )
                except Exception as err:
                    _LOGGER.error("Ошибка обновления журнала чата с ответом помощника: %s", err)
            else:
                self.history[conversation_id].extend([
                    {"role": "user", "content": user_input.text},
                    {"role": "assistant", "content": final_content},
                ])
                max_history = settings["chat_history"] * 2
                if len(self.history[conversation_id]) > max_history:
                    self.history[conversation_id] = self.history[conversation_id][-max_history:]
                    _LOGGER.debug("Обрезана история для conversation_id=%s до %s сообщений", 
                                  conversation_id, max_history)

            if LLM_AVAILABLE and chat_log:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_speech(final_content or "")
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                )
            else:
                return conversation.ConversationResult(
                    response=ConversationResponse(
                        speech={"plain": {"speech": final_content or ""}}
                    ),
                    conversation_id=conversation_id,
                )