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
                # При успешной проверке ключа, сразу получаем список моделей для первого шага
                models = await self.hass.async_add_executor_job(client.models.list)
                model_ids = [model.id for model in models.data]
                LOGGER.debug("Доступные модели Mistral: %s", model_ids)

                # Сохраняем клиента в runtime_data для использования в опциях
                self.hass.data.setdefault(DOMAIN, {})[self.context["flow_id"]] = client

                # Переходим к шагу выбора модели после успешной проверки ключа
                return await self.async_step_model_select(model_ids=model_ids, api_key=user_input[CONF_API_KEY])

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

    async def async_step_model_select(self, user_input=None, model_ids=None, api_key=None):
        """Шаг выбора модели из списка."""
        errors = {}
        if user_input is not None:
            # Модель выбрана, создаем запись конфигурации
            return self.async_create_entry(
                title="Mistral AI Conversation",
                data={
                    CONF_API_KEY: api_key, # Используем API ключ, полученный на предыдущем шаге
                    CONF_MODEL: user_input[CONF_MODEL],
                },
            )

        # Если model_ids не были переданы (например, при перезагрузке потока),
        # нужно снова получить клиента и список моделей.
        if model_ids is None:
             try:
                # Пытаемся получить клиента из runtime_data, сохраненного на шаге user
                client = self.hass.data.get(DOMAIN, {}).get(self.context["flow_id"])
                if client is None:
                     # Если клиента нет, возможно, поток начат не с шага user, или что-то пошло не так.
                     # В этом случае, возможно, лучше вернуться на шаг user или показать ошибку.
                     # Для простоты сейчас попробуем получить клиента заново, если API ключ доступен.
                     api_key = self.init_data.get(CONF_API_KEY) # Пытаемся получить ключ из init_data
                     if api_key:
                         def _init_client():
                              return Mistral(api_key=api_key)
                         client = await self.hass.async_add_executor_job(_init_client)
                         self.hass.data.setdefault(DOMAIN, {})[self.context["flow_id"]] = client # Сохраняем клиента
                     else:
                         LOGGER.error("Не удалось получить клиента Mistral для шага выбора модели.")
                         errors["base"] = "unknown_error"
                         # Возвращаемся на первый шаг, если не удалось получить API ключ
                         return await self.async_step_user()


                models = await self.hass.async_add_executor_job(client.models.list)
                model_ids = [model.id for model in models.data]
                LOGGER.debug("Доступные модели Mistral (получено на шаге выбора модели): %s", model_ids)

             except Exception as err:
                 LOGGER.error("Ошибка получения списка моделей Mistral на шаге выбора модели: %s", err)
                 errors["base"] = "connection_error" # Или другая более специфичная ошибка
                 # Если не удалось получить список моделей, показываем ошибку и остаемся на этом шаге
                 # или возвращаемся на шаг user, если проблема с ключом
                 if "authentication" in str(err).lower() or "api key" in str(err).lower():
                     errors["base"] = "auth_error"
                     return await self.async_step_user() # Вернуться на шаг user при ошибке аутентификации


        if not model_ids:
             errors["base"] = "no_models_available" # Если список моделей пуст
             LOGGER.error("Список моделей от Mistral API пуст.")


        data_schema = vol.Schema({
            vol.Required(CONF_MODEL, default=DEFAULT_MODEL if DEFAULT_MODEL in model_ids else (model_ids[0] if model_ids else None)): vol.In(model_ids if model_ids else []),
        })

        return self.async_show_form(
            step_id="model_select",
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
                # Проверка выбранной модели при сохранении опций
                client = self.config_entry.runtime_data
                models = await self.hass.async_add_executor_job(client.models.list)
                model_ids = [model.id for model in models.data]
                if user_input[CONF_MODEL] not in model_ids:
                    errors["base"] = "model_unavailable"
                else:
                    # Успешно сохраняем опции
                    return self.async_create_entry(title="", data=user_input)
            except Exception as err:
                LOGGER.error("Ошибка проверки модели Mistral при сохранении опций: %s", err)
                errors["base"] = "connection_error" # Или другая более специфичная ошибка

        # Получаем список моделей для выпадающего списка в опциях
        model_ids = []
        try:
            client = self.config_entry.runtime_data
            models = await self.hass.async_add_executor_job(client.models.list)
            model_ids = [model.id for model in models.data]
            LOGGER.debug("Доступные модели Mistral (получено для опций): %s", model_ids)
        except Exception as err:
            LOGGER.error("Ошибка получения списка моделей Mistral для опций: %s", err)
            errors["base"] = "connection_error" # Показываем ошибку, но все равно пытаемся отобразить форму

        # Получаем список доступных LLM API для опций
        llm_api_options = ["none"]
        if LLM_AVAILABLE:
            try:
                LOGGER.debug("Attempting to get LLM APIs for options")
                llm_apis = llm.async_get_apis(self.hass)
                LOGGER.debug("Received LLM APIs for options: %s", llm_apis)

                # Ensure llm_apis is an iterable
                if not isinstance(llm_apis, list):
                     LOGGER.debug("LLM APIs for options is not a list, attempting to await")
                     # If not a list, assume it's an awaitable and await it
                     llm_apis = await llm_apis
                     LOGGER.debug("Awaited LLM APIs for options: %s", llm_apis)

                valid_apis = []
                # Используем try...except AttributeError для обработки объектов без атрибута 'id'
                for i, api in enumerate(llm_apis):
                    LOGGER.debug("Processing LLM API object at index %d for options", i)
                    try:
                        # Log the type of the current API object
                        LOGGER.debug("Current LLM API object type for options: %s", type(api).__name__)
                        # Пытаемся получить атрибут 'id' и добавить его, если успешно
                        LOGGER.debug("Checking for 'id' attribute")
                        api_id = api.name # Используем api.id
                        LOGGER.debug("'id' attribute found: %s", api_id)
                        valid_apis.append(api_id)
                        LOGGER.debug("Added API ID: %s to valid_apis for options", api_id)
                    except AttributeError:
                        # Логируем предупреждение, если у объекта нет ожидаемого атрибута 'id'
                        LOGGER.warning("LLM API object at index %d for options, type %s, does not have 'id' attribute", i, type(api).__name__)
                    except Exception as inner_err:
                        # Логируем любую другую неожиданную ошибку при обработке одного объекта API
                        LOGGER.error("Unexpected error processing LLM API object at index %d for options, type %s: %s", i, type(api).__name__, inner_err)


                llm_api_options.extend(valid_apis)
                LOGGER.debug("Available LLM APIs after filtering for options: %s", llm_api_options)

            except Exception as err:
                # Этот внешний except перехватывает ошибки от llm.async_get_apis
                LOGGER.warning("Ошибка получения LLM API для опций: %s", err)
                # Ошибку уже добавили выше, здесь просто логируем

        # Определяем схему данных для формы опций
        data_schema = vol.Schema({
            # Используем vol.In для создания выпадающего списка моделей
            vol.Optional(CONF_MODEL, default=self.config_entry.options.get(CONF_MODEL, DEFAULT_MODEL if DEFAULT_MODEL in model_ids else (model_ids[0] if model_ids else None))): vol.In(model_ids if model_ids else []),
            vol.Optional(CONF_TEMPERATURE, default=self.config_entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)): vol.Coerce(float),
            vol.Optional(CONF_TOP_P, default=self.config_entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)): vol.Coerce(float),
            vol.Optional(CONF_MAX_TOKENS, default=self.config_entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)): vol.Coerce(int),
            vol.Optional(CONF_CHAT_HISTORY, default=self.config_entry.options.get(CONF_CHAT_HISTORY, DEFAULT_CHAT_HISTORY)): vol.Coerce(int),
            # Делаем поле промпта многострочным
            vol.Optional(CONF_PROMPT, default=self.config_entry.options.get(CONF_PROMPT, DEFAULT_PROMPT), description={"multiline": True}): str,
            vol.Optional(CONF_LLM_HASS_API, default=self.config_entry.options.get(CONF_LLM_HASS_API, "none")): vol.In(llm_api_options),
        })

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
            errors=errors,
        )
