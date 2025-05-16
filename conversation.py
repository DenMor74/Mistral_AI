"""Платформа разговоров Mistral AI для Home Assistant."""
import logging
import asyncio
import json
import codecs # Импортируем codecs для escape_decode
from typing import Any, Literal # Импортируем Literal
import ulid # Импортируем ulid для генерации ID

# Импортируем LOGGER из const первым
from .const import (
    DOMAIN,
    LOGGER, # Используем LOGGER из const
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
    CONF_LLM_HASS_API, # Также импортируем LLM_HASS_API здесь
)

# Импортируем HomeAssistant, ConfigEntry и AddEntitiesCallback
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry # Импортируем ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback # Импортируем AddEntitiesCallback
from homeassistant.exceptions import HomeAssistantError, ConfigEntryAuthFailed, ConfigEntryNotReady # Импортируем необходимые исключения

# Импортируем специфические типы из conversation
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
    AssistantContent, # Добавляем импорт AssistantContent
    ToolResultContent, # Добавляем импорт ToolResultContent
    ConversationEntityFeature, # Явно импортируем ConversationEntityFeature
)

# Импортируем conversation как ha_conversation на верхнем уровне
from homeassistant.components import conversation as ha_conversation


# _LOGGER = logging.getLogger(__name__) # Локальный логгер, если нужен, но лучше использовать LOGGER из const

# Импортируем convert из voluptuous_openapi
try:
    from voluptuous_openapi import convert
    VOLUPTUOUS_OPENAPI_AVAILABLE = True
    LOGGER.debug("Библиотека voluptuous_openapi доступна") # Используем LOGGER из const
except ImportError:
    VOLUPTUOUS_OPENAPI_AVAILABLE = False
    LOGGER.warning("Библиотека voluptuous_openapi недоступна. Форматирование схем инструментов может быть неполным.") # Используем LOGGER из const


try:
    from homeassistant.helpers import intent, llm
    LLM_AVAILABLE = True
    LOGGER.debug("Интеграция LLM доступна") # Используем LOGGER из const
except ImportError:
    LLM_AVAILABLE = False
    LOGGER.warning("Интеграция LLM недоступна; поддержка llm_hass_api будет отключена") # Используем LOGGER из const

# Максимальное количество итераций для вызовов инструментов
MAX_TOOL_ITERATIONS = 10

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Настройка сущности разговоров Mistral AI."""
    LOGGER.debug("Попытка настройки сущности разговоров Mistral AI") # Используем LOGGER из const
    client = entry.runtime_data
    try:
        # Проверка доступности выбранной модели при загрузке интеграции
        models = await hass.async_add_executor_job(client.models.list)
        model_ids = [model.id for model in models.data]
        selected_model = entry.options.get(CONF_MODEL, entry.data.get(CONF_MODEL, DEFAULT_MODEL)) # Проверяем опции, затем данные
        if selected_model not in model_ids:
            LOGGER.error("Модель %s недоступна. Доступные модели: %s", selected_model, model_ids) # Используем LOGGER из const
            raise ConfigEntryNotReady(f"Модель {selected_model} недоступна")
        LOGGER.debug("Клиент Mistral AI успешно инициализирован, доступные модели: %s", model_ids) # Используем LOGGER из const
    except Exception as err:
        LOGGER.error("Ошибка инициализации клиента Mistral AI: %s", err) # Используем LOGGER из const
        # Проверяем ошибки аутентификации/подключения
        if "authentication" in str(err).lower() or "api key" in str(err).lower():
             raise ConfigEntryAuthFailed("Неверный ключ API Mistral") from err
        if "connection" in str(err).lower() or "timeout" in str(err).lower():
             raise ConfigEntryNotReady(f"Ошибка соединения с Mistral API: {err}") from err
        raise ConfigEntryNotReady(f"Ошибка инициализации клиента Mistral AI: {err}") from err


    entity = MistralConversationEntity(hass, entry, client)
    async_add_entities([entity])
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entity
    LOGGER.debug("Успешная регистрация сущности разговоров Mistral AI с ID: %s", entity.entity_id) # Используем LOGGER из const

class ConversationResponse:
    """Временный класс для имитации ConversationResponse с методом as_dict."""
    def __init__(self, speech: dict[str, Any]) -> None:
        self.speech = speech
    def as_dict(self) -> dict[str, Any]:
        return {"speech": self.speech}

SUPPORTED_SCHEMA_KEYS = {
    # Ключи схемы, поддерживаемые Mistral API (на основе документации и сравнения)
    "type",
    "description",
    "enum",
    "properties",
    "required",
    "items",
    # Добавляем format, хотя его поддержка может варьироваться
    "format",
    # Добавляем nullable, хотя его поддержка может варьироваться
    "nullable",
    # Добавляем min_items, max_items, хотя их поддержка может варьироваться
    "min_items",
    "max_items",
}

# Функция для преобразования camelCase в snake_case (как в Google примере)
def _camel_to_snake(name: str) -> str:
    """Convert camel case to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def _format_schema(schema: Any) -> dict[str, Any]:
    """Форматирование схемы для совместимости с Mistral API."""
    # **Улучшение:** Проверяем, является ли схема словарем
    if not isinstance(schema, dict):
        LOGGER.warning("Схема инструмента не является словарем, имеет тип: %s. Возвращаем пустую схему.", type(schema).__name__) # Используем LOGGER из const
        return {} # Возвращаем пустой словарь, если схема не словарь

    # **Улучшение:** Обработка allOf (как в Google примере)
    if subschemas := schema.get("allOf"):
        if isinstance(subschemas, list):
            for subschema in subschemas:
                if isinstance(subschema, dict) and "type" in subschema:
                    # Рекурсивно форматируем первый под-схему с типом
                    return _format_schema(subschema)
            # Если ни одна под-схема не имеет типа, возвращаем форматирование первой под-схемы
            if subschemas and isinstance(subschemas[0], dict):
                 return _format_schema(subschemas[0])
            else:
                 LOGGER.warning("allOf содержит невалидные под-схемы: %s. Возвращаем пустую схему.", subschemas) # Используем LOGGER из const
                 return {}
        else:
            LOGGER.warning("allOf в схеме не является списком, имеет тип: %s. Пропускаем allOf.", type(subschemas).__name__) # Используем LOGGER из const


    result = {}
    # Добавляем try-except вокруг итерации по элементам схемы
    try:
        for key, val in schema.items():
            # **Улучшение:** Преобразуем ключ из camelCase в snake_case
            key = _camel_to_snake(key)

            if key not in SUPPORTED_SCHEMA_KEYS:
                continue
            try: # Добавляем try...except для обработки ошибок при обработке каждого ключа
                processed_val = val # Используем временную переменную для обработанного значения

                if key == "type":
                    # Mistral API ожидает типы в верхнем регистре
                    processed_val = str(val).upper()
                elif key == "format":
                    # **Улучшение:** Более строгая проверка формата (как в Google примере, адаптировано)
                    if schema.get("type") == "string" and val not in ("enum", "date-time"):
                        LOGGER.debug("Пропускаем формат '%s' для типа 'string'", val) # Используем LOGGER из const
                        continue # Пропускаем неподдерживаемые форматы для строк
                    if schema.get("type") == "number" and val not in ("float", "double"):
                        LOGGER.debug("Пропускаем формат '%s' для типа 'number'", val) # Используем LOGGER из const
                        continue # Пропускаем неподдерживаемые форматы для чисел
                    if schema.get("type") == "integer" and val not in ("int32", "int64"):
                        LOGGER.debug("Пропускаем формат '%s' для типа 'integer'", val) # Используем LOGGER из const
                        continue # Пропускаем неподдерживаемые форматы для целых чисел
                    if schema.get("type") not in ("string", "number", "integer"):
                         LOGGER.debug("Пропускаем формат '%s' для типа '%s'", val, schema.get("type")) # Используем LOGGER из const
                         continue # Пропускаем форматы для других типов
                elif key == "items":
                    # **Улучшение:** Проверяем тип val перед рекурсивным вызовом
                    if isinstance(val, dict):
                         # Добавляем try-except вокруг рекурсивного вызова
                         try:
                            processed_val = _format_schema(val) # Recursive call
                         except Exception as items_err:
                            LOGGER.error("Ошибка форматирования элемента 'items' в схеме: %s", items_err) # Используем LOGGER из const
                            continue # Skip items if formatting fails
                    else:
                        LOGGER.warning("Элемент 'items' в схеме не является словарем, имеет тип: %s. Пропускаем.", type(val).__name__) # Используем LOGGER из const
                        continue # Пропускаем этот ключ, если val не словарь
                elif key == "properties":
                    # Рекурсивно форматируем свойства
                    # **Улучшение:** Проверяем тип val перед рекурсивным форматированием свойств
                    if isinstance(val, dict):
                        # Добавляем try-except вокруг словаря comprehension
                        try:
                            processed_val = {k: _format_schema(v) for k, v in val.items()}
                        except Exception as prop_err:
                            LOGGER.error("Ошибка форматирования свойств в схеме: %s", prop_err) # Используем LOGGER из const
                            continue # Skip properties if formatting fails
                    else:
                        LOGGER.warning("Элемент 'properties' в схеме не является словарем, имеет тип: %s. Пропускаем.", type(val).__name__) # Используем LOGGER из const
                        continue # Пропускаем этот ключ, если val не словарь

                # Добавляем поле description, если оно есть в исходной схеме
                # **Изменение:** Проверяем, является ли schema словарем перед доступом по ключу 'description'
                if isinstance(schema, dict) and "description" in schema and key == "description":
                     result["description"] = schema["description"]

                # Добавляем try-except вокруг присваивания в result
                try:
                    result[key] = processed_val
                except Exception as assign_err:
                    LOGGER.error("Ошибка присваивания ключа '%s' со значением типа '%s' в форматированной схеме: %s", key, type(processed_val).__name__, assign_err) # Используем LOGGER из const
                    # Если присваивание не удалось, пропускаем этот ключ
                    continue

            except Exception as inner_err:
                LOGGER.error("Ошибка при обработке ключа '%s' в схеме инструмента: %s", key, inner_err) # Используем LOGGER из const
                continue # Пропускаем этот ключ в случае ошибки
    except Exception as iter_err:
         LOGGER.error("Ошибка при итерации по элементам схемы: %s", iter_err) # Используем LOGGER из const
         return {} # Возвращаем пустой словарь, если итерация не удалась


    # Преобразование enum в строки, если тип не STRING
    # Проверяем, что result является словарем перед доступом по ключу
    if isinstance(result, dict) and result.get("enum") and result.get("type") != "STRING":
        # **Улучшение:** Проверяем, является ли enum списком перед итерацией
        if isinstance(result["enum"], list):
            result["type"] = "STRING"
            # Добавляем try-except вокруг преобразования элементов enum
            try:
                result["enum"] = [str(item) for item in result["enum"]]
            except Exception as enum_err:
                LOGGER.error("Ошибка преобразования элементов enum в строки: %s", enum_err) # Используем LOGGER из const
                # В случае ошибки, удаляем enum или оставляем как есть (зависит от желаемого поведения)
                del result["enum"] # Удаляем некорректный enum
        else:
            LOGGER.warning("Элемент 'enum' в схеме не является списком, имеет тип: %s. Пропускаем преобразование.", type(result["enum"]).__name__) # Используем LOGGER из const

    # **Улучшение:** Обработка объекта без свойств (как в Google примере)
    if isinstance(result, dict) and result.get("type") == "OBJECT" and not result.get("properties"):
         LOGGER.warning("Объект в схеме без свойств. Добавляем свойство 'json' типа STRING в качестве запасного варианта.") # Используем LOGGER из const
         result["properties"] = {"json": {"type": "STRING"}}
         result["required"] = [] # Объект без свойств не требует ничего по умолчанию

    return result # Возвращаем result, который теперь dict

# Функция для escape-декодирования (как в Google примере)
def _escape_decode(value: Any) -> Any:
    """Recursively call codecs.escape_decode on all values."""
    if isinstance(value, str):
        try:
            # Декодируем только если это строка
            return codecs.escape_decode(bytes(value, "utf-8"))[0].decode("utf-8")  # type: ignore[attr-defined]
        except Exception as err:
            LOGGER.warning("Ошибка escape-декодирования строки: %s. Возвращаем исходную строку.", err) # Используем LOGGER из const
            return value # Возвращаем исходное значение при ошибке
    if isinstance(value, list):
        return [_escape_decode(item) for item in value]
    if isinstance(value, dict):
        return {k: _escape_decode(v) for k, v in value.items()}
    return value

def _format_tool(tool: llm.Tool, chat_log: ChatLog | None) -> dict[str, Any] | None: # Используем ChatLog из явного импорта, возвращаем dict или None
    """Форматирование спецификации инструмента. Возвращает отформатированный инструмент или None в случае ошибки."""
    # Проверяем, что tool является экземпляром llm.Tool
    if not isinstance(tool, llm.Tool):
        LOGGER.warning("Неожиданный тип объекта инструмента: %s. Ожидается llm.Tool. Пропускаем.", type(tool).__name__) # Используем LOGGER из const
        return None # Возвращаем None, если объект не является инструментом

    parameters = {}
    # Проверяем, что tool.parameters не None и имеет атрибут schema
    if tool.parameters and hasattr(tool.parameters, 'schema') and tool.parameters.schema is not None: # Добавлена проверка на None
        try:
            # **Изменение:** Используем voluptuous_openapi.convert перед форматированием схемы
            if VOLUPTUOUS_OPENAPI_AVAILABLE and chat_log and chat_log.llm_api and hasattr(chat_log.llm_api, 'custom_serializer'):
                LOGGER.debug("Используем voluptuous_openapi.convert для инструмента '%s'", tool.name) # Используем LOGGER из const
                # Используем custom_serializer из llm_api, если доступен
                converted_schema = convert(tool.parameters, custom_serializer=chat_log.llm_api.custom_serializer)
            elif VOLUPTUOUS_OPENAPI_AVAILABLE:
                 LOGGER.debug("Используем voluptuous_openapi.convert без custom_serializer для инструмента '%s'", tool.name) # Используем LOGGER из const
                 converted_schema = convert(tool.parameters)
            else:
                 LOGGER.warning("voluptuous_openapi недоступен или custom_serializer отсутствует. Форматирование схемы для инструмента '%s' может быть неполным.", tool.name) # Используем LOGGER из const
                 converted_schema = tool.parameters.schema # Используем исходную схему как запасной вариант

            # Добавляем try-except вокруг вызова _format_schema
            try:
                parameters = _format_schema(converted_schema)
                LOGGER.debug("Схема инструмента '%s' после форматирования: %s", tool.name, parameters) # Используем LOGGER из const
            except Exception as format_schema_err:
                LOGGER.error("Ошибка в _format_schema для инструмента '%s': %s", tool.name, format_schema_err) # Используем LOGGER из const
                parameters = {} # В случае ошибки форматирования схемы, используем пустые параметры

        except Exception as err: # Этот except теперь должен ловить только ошибки от voluptuous_openapi.convert
            LOGGER.error("Ошибка конвертации схемы для инструмента '%s' (voluptuous_openapi.convert): %s", tool.name, err) # Используем LOGGER из const
            # В случае ошибки конвертации, возвращаем пустые параметры,
            # чтобы не сломать весь список инструментов
            parameters = {}
    elif tool.parameters and not hasattr(tool.parameters, 'schema'):
         LOGGER.warning("Инструмент '%s' имеет параметры, но отсутствует атрибут 'schema'", tool.name) # Используем LOGGER из const
    elif not tool.parameters:
         LOGGER.debug("Инструмент '%s' не имеет параметров", tool.name) # Используем LOGGER из const


    # Проверяем, что параметры являются словарем перед возвратом
    if not isinstance(parameters, dict):
         LOGGER.error("Отформатированные параметры для инструмента '%s' не являются словарем, имеют тип: %s. Пропускаем инструмент.", tool.name, type(parameters).__name__) # Используем LOGGER из const
         return None # Пропускаем инструмент, если параметры не словарь

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters, # This is the formatted schema
        },
    }

# Импортируем Mistral непосредственно перед использованием в классе
from mistralai import Mistral

class MistralConversationEntity(ConversationEntity): # Используем ConversationEntity из явного импорта
    """Сущность разговоров Mistral AI."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, client: Mistral) -> None:
        """Инициализация сущности."""
        self.hass = hass
        self.entry = entry
        self._client = client
        # Используем словарь для хранения истории по conversation_id, если ChatLog недоступен
        self.history: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}"
        self._attr_entity_id = f"conversation.{DOMAIN}_{entry.entry_id}" # Уникальный entity_id
        self._attr_name = "Mistral AI Conversation"
        self._attr_has_entity_name = True
        self.entity_id = self._attr_entity_id # Устанавливаем entity_id здесь
        self._attr_should_poll = False
        self._is_ready = False # Флаг готовности клиента Mistral

        # Определяем поддерживаемые возможности на основе наличия LLM API в опциях
        selected_llm_api = entry.options.get(CONF_LLM_HASS_API)
        self._attr_supported_features = (
            ConversationEntityFeature.CONTROL # Используем ConversationEntityFeature из явного импорта
            if LLM_AVAILABLE and selected_llm_api and selected_llm_api != "none"
            else 0
        )
        LOGGER.debug("Инициализирована MistralConversationEntity с entity_id: %s, unique_id: %s, llm_available: %s, supported_features: %s", # Используем LOGGER из const
                      self._attr_entity_id, self._attr_unique_id, LLM_AVAILABLE, self._attr_supported_features)


    async def async_added_to_hass(self) -> None:
        """Выполняется при добавлении сущности в Home Assistant."""
        LOGGER.debug("Добавление сущности %s в Home Assistant", self.entity_id) # Используем LOGGER из const
        # Проверяем готовность клиента при добавлении сущности
        if self._client is None:
             LOGGER.error("Клиент Mistral не инициализирован для сущности %s", self.entity_id) # Используем LOGGER из const
             self._is_ready = False
             return # Не регистрируем агент, если клиент не готов

        try:
            # Проверяем доступность выбранной модели при добавлении сущности
            models = await self.hass.async_add_executor_job(self._client.models.list)
            model_ids = [model.id for model in models.data]
            selected_model = self.entry.options.get(CONF_MODEL, self.entry.data.get(CONF_MODEL, DEFAULT_MODEL))
            if selected_model not in model_ids:
                LOGGER.error("Модель %s недоступна для сущности %s. Доступные модели: %s", # Используем LOGGER из const
                              selected_model, self.entity_id, model_ids)
                self._is_ready = False
                # Не вызываем исключение здесь, чтобы сущность добавилась, но будет недоступна
                return
            LOGGER.debug("Клиент Mistral AI успешно инициализирован для сущности %s, доступные модели: %s", # Используем LOGGER из const
                          self.entity_id, model_ids)
            self._is_ready = True
            # Регистрируем агент разговора только если клиент готов и модель доступна
            # Используем ha_conversation из импорта верхнего уровня
            ha_conversation.async_set_agent(self.hass, self.entry, self)
            LOGGER.debug("Агент %s успешно зарегистрирован", self.entity_id) # Используем LOGGER из const
        except Exception as err:
            # Логируем ошибку, но не вызываем исключение, чтобы сущность добавилась
            LOGGER.error("Ошибка инициализации клиента Mistral для сущности %s: %s", self.entity_id, err) # Используем LOGGER из const
            self._is_ready = False


    async def async_will_remove_from_hass(self) -> None:
        """Выполняется при удалении сущности из Home Assistant."""
        LOGGER.debug("Удаление сущности %s из Home Assistant", self.entity_id) # Используем LOGGER из const
        # Убираем агент разговора при удалении сущности
        # Используем ha_conversation из импорта верхнего уровня
        ha_conversation.async_unset_agent(self.hass, self.entry)
        # Очищаем историю для этого conversation_id при удалении сущности (опционально)
        # for conv_id in list(self.history.keys()):
        #     del self.history[conv_id]
        await super().async_will_remove_from_hass()

    @property
    def is_ready(self) -> bool:
        """Проверка готовности сущности."""
        # Сущность готова, если клиент инициализирован и флаг _is_ready установлен
        LOGGER.debug("Проверка is_ready для сущности %s: client=%s, _is_ready=%s", # Используем LOGGER из const
                      self.entity_id, self._client is not None, self._is_ready)
        return self._client is not None and self._is_ready

    @property
    def supported_languages(self) -> list[str] | Literal["*"]: # Добавляем Literal для поддержки "*"
        """Список поддерживаемых языков."""
        # У Mistral нет строгого списка поддерживаемых языков, указываем основные
        # Или возвращаем MATCH_ALL, если поддерживаются все языки
        # from homeassistant.const import MATCH_ALL # Может потребоваться импорт MATCH_ALL
        # return MATCH_ALL
        return ["en", "es", "fr", "de", "it", "ru", "zh", "ja", "ko"]

    async def async_process(self, user_input: ConversationInput) -> ConversationResult: # Используем ConversationInput и ConversationResult из явного импорта
        """Обработка пользовательского ввода и возврат ответа."""
        LOGGER.debug("Обработка ввода разговора для сущности %s: текст=%s, conversation_id=%s, agent_id=%s", # Используем LOGGER из const
                      self.entity_id, user_input.text, user_input.conversation_id, user_input.agent_id)

        # Проверяем готовность клиента перед обработкой
        if self._client is None or not self._is_ready:
            LOGGER.error("Клиент Mistral не готов или не инициализирован для обработки запроса сущности %s", self.entity_id) # Используем LOGGER из const
            # Пытаемся переинициализировать клиента, если он None
            if self._client is None and self.entry.runtime_data:
                self._client = self.entry.runtime_data
                LOGGER.debug("Клиент Mistral восстановлен из runtime_data") # Используем LOGGER из const
                # Повторно проверяем готовность после восстановления клиента
                if not self._is_ready:
                    try:
                        await self.async_added_to_hass() # Повторно вызываем async_added_to_hass для проверки и регистрации
                    except Exception as err:
                         LOGGER.error("Ошибка при повторной проверке готовности после восстановления клиента: %s", err) # Используем LOGGER из const
                         raise HomeAssistantError(f"Невозможно обработать разговор: клиент не готов") from err

            if self._client is None or not self._is_ready:
                 raise HomeAssistantError(f"Невозможно обработать разговор: клиент Mistral AI не готов")


        conversation_id = user_input.conversation_id or ulid.ulid()
        chat_log: ChatLog | None = None # Используем ChatLog из явного импорта
        tools: list[dict[str, Any]] | None = None
        selected_llm_api = self.entry.options.get(CONF_LLM_HASS_API)
        llm_api_id_to_use = selected_llm_api if LLM_AVAILABLE and selected_llm_api and selected_llm_api != "none" else None

        # Инициализируем ChatLog и получаем инструменты только если выбран реальный LLM API
        if llm_api_id_to_use:
            LOGGER.debug("Попытка инициализации журнала чата LLM с llm_api_id: %s", llm_api_id_to_use) # Используем LOGGER из const
            try:
                # Инициализируем ChatLog с выбранным LLM API ID
                chat_log = ChatLog(self.hass, conversation_id, user_input.agent_id) # Используем ChatLog из явного импорта
                await chat_log.async_update_llm_data(
                    DOMAIN,
                    user_input,
                    llm_api_id_to_use, # Используем выбранный LLM API ID
                    self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
                )
                if chat_log.llm_api and chat_log.llm_api.tools: # Проверяем, что tools не пустой список
                    # **Улучшенное логирование:** Логируем исходные инструменты
                    LOGGER.debug("Исходные инструменты от LLM API %s (%d): %s", llm_api_id_to_use, len(chat_log.llm_api.tools), chat_log.llm_api.tools) # Используем LOGGER из const

                    # **Изменение:** Передаем chat_log в _format_tool для доступа к custom_serializer
                    # Добавляем try-except вокруг форматирования каждого инструмента
                    formatted_tools = []
                    for tool in chat_log.llm_api.tools:
                        try:
                            formatted_tool = _format_tool(tool, chat_log)
                            if formatted_tool: # Добавляем инструмент только если форматирование успешно
                                formatted_tools.append(formatted_tool)
                        except Exception as format_err:
                            LOGGER.error("Ошибка форматирования инструмента '%s': %s", tool.name if hasattr(tool, 'name') else 'Unknown Tool', format_err) # Используем LOGGER из const
                            # Если форматирование одного инструмента не удалось, просто пропускаем его

                    tools = formatted_tools # Присваиваем отформатированный список инструментов

                    # **Улучшенное логирование:** Логируем отформатированные инструменты как JSON
                    LOGGER.debug("Отформатированные инструменты для Mistral API (%d): %s", len(tools), json.dumps(tools, indent=2)) # Используем LOGGER из const

                    if not tools:
                        LOGGER.warning("Инструменты недоступны для LLM API ID: %s после форматирования", llm_api_id_to_use) # Используем LOGGER из const
                else:
                    LOGGER.warning("Экземпляр LLM API недоступен или список инструментов пуст для ID: %s после async_update_llm_data", llm_api_id_to_use) # Используем LOGGER из const
                    chat_log = None # Сбрасываем chat_log, если LLM API недоступен или нет инструментов
                    tools = None
            except Exception as err:
                # **Изменение:** Логируем ошибку с более конкретным сообщением
                LOGGER.error("Ошибка инициализации журнала чата LLM или форматирования инструментов для llm_api_id %s: %s", llm_api_id_to_use, err) # Используем LOGGER из const
                chat_log = None
                tools = None
        else:
            LOGGER.debug("LLM API для инструментов не выбран или недоступен (llm_api_id_to_use=%s); используется базовый разговор", # Используем LOGGER из const
                          llm_api_id_to_use)


        settings = {
            "model": self.entry.options.get(CONF_MODEL, self.entry.data.get(CONF_MODEL, DEFAULT_MODEL)), # Проверяем опции, затем данные
            "temperature": self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            "top_p": self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
            "max_tokens": self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            "chat_history": self.entry.options.get(CONF_CHAT_HISTORY, DEFAULT_CHAT_HISTORY),
            "prompt": self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
        }
        LOGGER.debug("Используемые настройки для сущности %s: %s", self.entity_id, settings) # Используем LOGGER из const

        async with self._lock:
            # Формируем список сообщений для отправки в Mistral API
            messages = []
            if settings["prompt"]:
                messages.append({"role": "system", "content": settings["prompt"]})

            # Добавляем историю чата
            if chat_log:
                # Используем историю из ChatLog, если она доступна
                for content in chat_log.content[-settings["chat_history"]:]:
                    # **Изменение:** Проверяем, что content не None перед доступом к content.content
                    if content and content.content is not None: # Также проверяем, что content.content не None
                        # Преобразуем роли из ChatLog в формат Mistral (user, assistant, tool)
                        # Роли в ChatLog могут быть 'user', 'assistant', 'tool', 'system'
                        # Mistral API ожидает 'user', 'assistant', 'tool', 'system'
                        # Проверяем, что роль поддерживается Mistral API
                        mistral_role = content.role
                        if mistral_role not in ["user", "assistant", "tool", "system"]:
                             LOGGER.warning("Неизвестная роль в истории чата ChatLog: %s. Пропускаем сообщение.", content.role) # Используем LOGGER из const
                             continue
                        # **Изменение:** Добавляем tool_calls к сообщениям с ролью 'assistant', если они есть
                        message_data = {"role": mistral_role, "content": content.content}
                        # **Изменение:** Проверяем, что content является AssistantContent перед доступом к content.tool_calls
                        if isinstance(content, AssistantContent) and content.tool_calls: # Используем AssistantContent из явного импорта
                             # Добавляем try-except вокруг обработки tool_calls из истории
                             try:
                                 message_data["tool_calls"] = [
                                     {
                                         "id": ulid.ulid(), # Генерируем ID для tool_call
                                         "function": {
                                             "name": tc.tool_name,
                                             "arguments": json.dumps(_escape_decode(tc.tool_args)), # Escape-deкодируем и сериализуем аргументы
                                         }
                                     } for tc in content.tool_calls if hasattr(tc, 'tool_name') and hasattr(tc, 'tool_args') # Добавляем проверки атрибутов
                                 ]
                                 LOGGER.debug("Добавлен tool_calls к сообщению помощника в истории: %s", message_data) # Используем LOGGER из const
                             except Exception as hist_tool_err:
                                 LOGGER.error("Ошибка обработки tool_calls из истории чата: %s", hist_tool_err) # Используем LOGGER из const
                                 # Если обработка tool_calls из истории не удалась, пропускаем их
                                 if "tool_calls" in message_data:
                                     del message_data["tool_calls"]

                        messages.append(message_data)

                LOGGER.debug("История чата сформирована из ChatLog: %s", messages) # Используем LOGGER из const
            else:
                # Если ChatLog недоступен, используем внутреннюю историю
                if conversation_id not in self.history:
                    self.history[conversation_id] = []
                    LOGGER.debug("Инициализирована новая внутренняя история для conversation_id=%s", conversation_id) # Используем LOGGER из const

                # Добавляем только сообщения с поддерживаемыми ролями из внутренней истории
                valid_history_messages = []
                for msg in self.history[conversation_id][-settings["chat_history"]:]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                         if msg["role"] in ["user", "assistant", "tool", "system"]:
                              valid_history_messages.append(msg)
                         else:
                              LOGGER.warning("Неизвестная роль во внутренней истории чата: %s. Пропускаем сообщение.", msg.get("role")) # Используем LOGGER из const
                    else:
                         LOGGER.warning("Неверный формат сообщения во внутренней истории чата: %s. Пропускаем.", msg) # Используем LOGGER из const

                messages.extend(valid_history_messages)
                LOGGER.debug("История чата сформирована из внутренней истории: %s", messages) # Используем LOGGER из const


            # Добавляем текущее сообщение пользователя
            messages.append({"role": "user", "content": user_input.text})
            LOGGER.debug("Финальные сообщения для API Mistral: %s", messages) # Используем LOGGER из const

            # Цикл для обработки вызовов инструментов
            final_content = None
            for iteration in range(MAX_TOOL_ITERATIONS):
                LOGGER.debug("Итерация %d для вызова API Mistral", iteration + 1) # Используем LOGGER из const
                try:
                    response = await self._client.chat.complete_async(
                        model=settings["model"],
                        messages=messages,
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        max_tokens=settings["max_tokens"],
                        # Передаем инструменты только если они были получены через ChatLog и успешно отформатированы
                        tools=tools if tools else None,
                        tool_choice="auto" if tools else None, # Используем auto, если есть инструменты
                    )

                    # **Улучшенное логирование:** Логируем полный ответ от API Mistral
                    LOGGER.debug("Полный ответ от API Mistral (итерация %d): %s", iteration + 1, response.model_dump_json(indent=2)) # Используем LOGGER из const


                    if not response.choices:
                        LOGGER.error("Нет вариантов ответа от API Mistral AI для сущности %s (итерация %d)", self.entity_id, iteration + 1) # Используем LOGGER из const
                        # Если это не первая итерация и нет вариантов, возможно, что-то пошло не так после вызова инструментов
                        if iteration > 0:
                             raise HomeAssistantError("Нет ответа от API Mistral AI после вызовов инструментов")
                         # Если это первая итерация и нет вариантов, возможно, проблема с моделью или запросом
                        raise HomeAssistantError("Нет ответа от API Mistral AI")


                    if response.usage:
                        # Логируем статистику использования токенов
                        LOGGER.debug("Статистика использования токенов (итерация %d): prompt=%d, completion=%d, total=%d", # Используем LOGGER из const
                                      iteration + 1, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
                        # Добавляем статистику в trace ChatLog, если доступен
                        if chat_log:
                             chat_log.async_trace({
                                "stats": {
                                    "input_tokens": response.usage.prompt_tokens,
                                    "output_tokens": response.usage.completion_tokens,
                                    "total_tokens": response.usage.total_tokens,
                                }
                             })


                except Exception as err:
                    LOGGER.error("Ошибка API Mistral для сущности %s (итерация %d): %s", self.entity_id, iteration + 1, err) # Используем LOGGER из const
                    raise HomeAssistantError(f"Ошибка связи с Mistral AI: {err}") from err

                # Обрабатываем вызовы инструментов только если ChatLog и инструменты доступны И модель вернула tool_calls
                if chat_log and tools and response.choices[0].message.tool_calls:
                    LOGGER.debug("Обработка вызовов инструментов (итерация %d): %s", iteration + 1, response.choices[0].message.tool_calls) # Используем LOGGER из const
                    tool_messages = [] # Список сообщений с результатами инструментов для добавления в историю
                    # Добавляем сообщение помощника с вызовами инструментов в историю для следующей итерации
                    messages.append(response.choices[0].message.model_dump())

                    for tool_call in response.choices[0].message.tool_calls:
                        # Добавляем try-except для обработки каждого отдельного вызова инструмента
                        try:
                            tool_name = tool_call.function.name
                            tool_args_str = tool_call.function.arguments
                            tool_call_id = tool_call.id

                            LOGGER.debug("Вызов инструмента '%s' с аргументами (строка): %s (итерация %d)", tool_name, tool_args_str, iteration + 1) # Используем LOGGER из const

                            try:
                                # Пытаемся распарсить аргументы как JSON
                                tool_args = json.loads(tool_args_str)
                            except json.JSONDecodeError as err:
                                LOGGER.error("Неверные аргументы инструмента для '%s': %s (итерация %d)", tool_name, err, iteration + 1) # Используем LOGGER из const
                                # Добавляем сообщение об ошибке парсинга в историю как результат инструмента
                                error_message = {"error": f"Неверный формат аргументов инструмента: {err}"}
                                tool_messages.append(
                                    {
                                        "role": "tool",
                                        "content": json.dumps(error_message), # Отправляем JSON строку ошибки
                                        "tool_call_id": tool_call_id,
                                    }
                                )
                                continue # Переходим к следующему вызову инструмента

                            LOGGER.debug("Вызов инструмента '%s' с аргументами (JSON): %s (итерация %d)", tool_name, tool_args, iteration + 1) # Используем LOGGER из const
                            try:
                                # Выполняем инструмент через ChatLog
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
                                LOGGER.debug("Результат инструмента для '%s' (итерация %d): %s", tool_name, iteration + 1, tool_result) # Используем LOGGER из const
                                # Добавляем результат выполнения инструмента в список сообщений
                                tool_messages.append(
                                    {
                                        "role": "tool",
                                        "content": json.dumps(tool_result), # Отправляем JSON строку результата
                                        "tool_call_id": tool_call_id,
                                    }
                                )
                            except Exception as err:
                                LOGGER.error("Ошибка выполнения инструмента '%s' (итерация %d): %s", tool_name, iteration + 1, err) # Используем LOGGER из const
                                # Добавляем сообщение об ошибке выполнения в историю как результат инструмента
                                error_message = {"error": f"Ошибка выполнения инструмента '{tool_name}': {err}"}
                                tool_messages.append(
                                    {
                                        "role": "tool",
                                        "content": json.dumps(error_message), # Отправляем JSON строку ошибки
                                        "tool_call_id": tool_call_id,
                                    }
                                )
                        except Exception as single_tool_call_err:
                             LOGGER.error("Неожиданная ошибка при обработке отдельного вызова инструмента: %s", single_tool_call_err) # Используем LOGGER из const
                             # Если при обработке отдельного вызова инструмента произошла неожиданная ошибка, пропускаем его

                    # Добавляем результаты выполнения инструментов в историю для следующей итерации
                    messages.extend(tool_messages)
                    LOGGER.debug("Сообщения после обработки инструментов (итерация %d): %s", iteration + 1, messages) # Используем LOGGER из const

                    # Продолжаем цикл для получения финального ответа от модели
                    continue # Переходим к следующей итерации с обновленными сообщениями

                # Если нет вызовов инструментов в ответе И есть текстовый контент, это финальный ответ
                final_content = response.choices[0].message.content
                if final_content is not None: # Проверяем, что контент не None
                    if not isinstance(final_content, str):
                        LOGGER.error("Неверный тип текстового контента ответа для сущности %s: %s", self.entity_id, type(final_content)) # Используем LOGGER из const
                        # Если ответ не строка, но есть tool_calls, возможно, это только tool_calls без текста
                        if not (response.choices[0].message.tool_calls or response.choices[0].message.content):
                             # Если нет ни текста, ни tool_calls, это ошибка
                             raise HomeAssistantError("Неверный ответ от API Mistral AI: отсутствует текст или вызовы инструментов")

                        # Если есть tool_calls, но нет текста, считаем это не финальным ответом и продолжаем цикл
                        if response.choices[0].message.tool_calls:
                             LOGGER.debug("Ответ без текстового контента, но с tool_calls. Продолжаем итерации.") # Используем LOGGER из const
                             continue # Продолжаем цикл, даже если final_content None, если были tool_calls
                        else:
                             # Если нет ни текста, ни tool_calls, но итерации не исчерпаны, возможно, модель просто молчит?
                             # В таком случае, считаем это пустым ответом и выходим из цикла.
                             LOGGER.warning("Ответ без текстового контента и без tool_calls. Завершаем итерации с пустым ответом.") # Используем LOGGER из const
                             final_content = "" # Устанавливаем пустой контент
                             break # Выходим из цикла


                    LOGGER.debug("Финальный ответ Mistral AI: %s", final_content) # Используем LOGGER из const
                    break # Выходим из цикла, если получен финальный ответ (строка)

                # Если нет вызовов инструментов И нет текстового контента, но итерации не исчерпаны
                LOGGER.warning("Ответ без текстового контента и без tool_calls. Завершаем итерации.") # Используем LOGGER из const
                final_content = "" # Устанавливаем пустой контент
                break # Выходим из цикла

            else:
                # Если цикл завершился без break (превышен лимит итераций)
                LOGGER.error("Превышен лимит итераций вызовов инструментов для сущности %s", self.entity_id) # Используем LOGGER из const
                raise HomeAssistantError("Превышен лимит итераций вызовов инструментов")

            # Сохраняем историю чата
            if chat_log:
                try:
                    # Добавляем сообщение пользователя и финальный ответ помощника в ChatLog
                    # ChatLog.async_update_llm_data уже добавил сообщение пользователя
                    # Сообщения с tool_calls и tool_results уже добавлены в messages в цикле
                    # Теперь добавляем только финальный текстовый ответ, если он есть
                    if final_content is not None: # Убеждаемся, что final_content не None
                         await chat_log.async_add_content(
                            role="assistant",
                            content=final_content,
                            agent_id=user_input.agent_id,
                        )
                    LOGGER.debug("История чата обновлена в ChatLog") # Используем LOGGER из const
                except Exception as err:
                    LOGGER.error("Ошибка обновления журнала чата с ответом помощника: %s", err) # Используем LOGGER из const
            else:
                # Если ChatLog недоступен, обновляем внутреннюю историю
                # Сообщение пользователя уже добавлено в messages перед вызовом API
                # Сообщения с tool_calls и tool_results уже добавлены в messages в цикле
                # Добавляем только финальный текстовый ответ, если он есть
                if final_content is not None:
                    self.history.setdefault(conversation_id, []).append({"role": "assistant", "content": final_content})

                # Обрезаем внутреннюю историю, чтобы не росла бесконечно
                max_history = settings["chat_history"] * 2 # Храним в 2 раза больше, чем используется для запроса
                if len(self.history.get(conversation_id, [])) > max_history:
                    self.history[conversation_id] = self.history[conversation_id][-max_history:]
                    LOGGER.debug("Обрезана внутренняя история для conversation_id=%s до %s сообщений", # Используем LOGGER из const
                                  conversation_id, max_history)
                LOGGER.debug("Внутренняя история чата обновлена") # Используем LOGGER из const


            # Формируем результат разговора
            if LLM_AVAILABLE and chat_log:
                # Используем IntentResponse, если ChatLog доступен
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_speech(final_content or "")
                LOGGER.debug("Возврат ConversationResult с IntentResponse (ChatLog доступен)") # Используем LOGGER из const
                return ha_conversation.ConversationResult( # Используем ha_conversation из homeassistant.components
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                )
            else:
                # Используем базовый ConversationResponse, если ChatLog недоступен
                LOGGER.debug("Возврат ConversationResult с базовым ConversationResponse (ChatLog недоступен)") # Используем LOGGER из const
                return ha_conversation.ConversationResult( # Используем ha_conversation из homeassistant.components
                    response=ConversationResponse(
                        speech={"plain": {"speech": final_content or ""}}
                    ),
                    conversation_id=conversation_id,
                )
