"""Поток конфигурации для интеграции Mistral AI Conversation."""
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.const import CONF_API_KEY
from homeassistant.helpers import config_validation as cv
from mistralai import Mistral

from .const import (
    DOMAIN,
    LOGGER,
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
    CONF_LLM_HASS_API,
)

try:
    from homeassistant.helpers import llm
    LLM_AVAILABLE = True
    LOGGER.debug("Интеграция LLM доступна для потока конфигурации")
except ImportError:
    LLM_AVAILABLE = False
    LOGGER.warning("Интеграция LLM недоступна для потока конфигурации; LLM API будет отключено")

class MistralAIConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Поток конфигурации для Mistral AI Conversation."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Обработка начального шага пользовательского ввода."""
        errors = {}
        if user_input is not None:
            try:
                # Проверка API ключа
                def _init_client():
                    return Mistral(api_key=user_input[CONF_API_KEY])

                client = await self.hass.async_add_executor_job(_init_client)
                models = await self.hass.async_add_executor_job(client.models.list)
                model_ids = [model.id for model in models.data]
                LOGGER.debug("Доступные модели Mistral: %s", model_ids)
                if DEFAULT_MODEL not in model_ids:
                    errors["base"] = "model_unavailable"
                else:
                    return self.async_create_entry(
                        title="Mistral AI Conversation",
                        data={CONF_API_KEY: user_input[CONF_API_KEY]},
                    )
            except Exception as err:
                LOGGER.error("Ошибка проверки API ключа Mistral: %s", err)
                errors["base"] = "auth_error" if "authentication" in str(err).lower() else "connection_error"

        data_schema = vol.Schema({
            vol.Required(CONF_API_KEY): str,
        })
        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Получение потока опций."""
        return MistralAIOptionsFlowHandler(config_entry)

class MistralAIOptionsFlowHandler(config_entries.OptionsFlow):
    """Поток опций для Mistral AI Conversation."""

    def __init__(self, config_entry):
        """Инициализация потока опций."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Обработка начального шага опций."""
        errors = {}
        if user_input is not None:
            try:
                # Проверка модели
                client = self.config_entry.runtime_data
                models = await self.hass.async_add_executor_job(client.models.list)
                model_ids = [model.id for model in models.data]
                if user_input[CONF_MODEL] not in model_ids:
                    errors["base"] = "model_unavailable"
                else:
                    return self.async_create_entry(title="", data=user_input)
            except Exception as err:
                LOGGER.error("Ошибка проверки модели Mistral: %s", err)
                errors["base"] = "connection_error"

        llm_api_options = ["none"]
        if LLM_AVAILABLE:
            try:
                # Попробуем получить LLM API без await, если функция синхронная
                llm_apis = llm.async_get_apis(self.hass)
                # Проверяем, является ли результат списком
                if isinstance(llm_apis, list):
                    llm_api_options.extend([api.api_id for api in llm_apis])
                else:
                    # Если результат асинхронный, используем await
                    llm_apis = await llm_apis
                    llm_api_options.extend([api.api_id for api in llm_apis])
                LOGGER.debug("Доступные LLM API: %s", llm_api_options)
            except Exception as err:
                LOGGER.warning("Ошибка получения LLM API: %s", err)
                llm_api_options = ["none"]

        data_schema = vol.Schema({
            vol.Optional(CONF_MODEL, default=self.config_entry.options.get(CONF_MODEL, DEFAULT_MODEL)): str,
            vol.Optional(CONF_TEMPERATURE, default=self.config_entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)): vol.Coerce(float),
            vol.Optional(CONF_TOP_P, default=self.config_entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)): vol.Coerce(float),
            vol.Optional(CONF_MAX_TOKENS, default=self.config_entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)): vol.Coerce(int),
            vol.Optional(CONF_CHAT_HISTORY, default=self.config_entry.options.get(CONF_CHAT_HISTORY, DEFAULT_CHAT_HISTORY)): vol.Coerce(int),
            vol.Optional(CONF_PROMPT, default=self.config_entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)): str,
            vol.Optional(CONF_LLM_HASS_API, default=self.config_entry.options.get(CONF_LLM_HASS_API, "none")): vol.In(llm_api_options),
        })
        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
            errors=errors,
        )