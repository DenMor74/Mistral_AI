"""Интеграция Mistral AI Conversation для Home Assistant."""
import logging
import pkg_resources
from mistralai import Mistral
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady, ConfigEntryError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, CONF_MODEL, DEFAULT_MODEL, LOGGER

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
PLATFORMS = (Platform.CONVERSATION,)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Настройка интеграции Mistral AI Conversation."""
    LOGGER.debug("Инициализация интеграции Mistral AI Conversation")
    try:
        mistralai_version = pkg_resources.get_distribution("mistralai").version
        LOGGER.debug("Версия mistralai: %s", mistralai_version)
    except pkg_resources.DistributionNotFound:
        LOGGER.error("Библиотека mistralai не установлена")
        raise ConfigEntryError("Библиотека mistralai не установлена")
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Настройка интеграции Mistral AI Conversation из записи конфигурации."""
    LOGGER.debug("Начало настройки записи конфигурации для Mistral AI Conversation")
    try:
        def _init_client() -> Mistral:
            LOGGER.debug("Создание клиента Mistral с ключом API")
            return Mistral(api_key=entry.data[CONF_API_KEY])

        client = await hass.async_add_executor_job(_init_client)
        models = await hass.async_add_executor_job(client.models.list)
        model_ids = [model.id for model in models.data]
        LOGGER.debug("Доступные модели Mistral: %s", model_ids)
        selected_model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
        if selected_model not in model_ids:
            LOGGER.error("Модель %s недоступна. Доступные модели: %s", selected_model, model_ids)
            raise ConfigEntryError(f"Модель {selected_model} недоступна")

    except Exception as err:
        if "authentication" in str(err).lower() or "api key" in str(err).lower():
            LOGGER.error("Неверный ключ API Mistral: %s", err)
            raise ConfigEntryAuthFailed("Неверный ключ API Mistral") from err
        if "connection" in str(err).lower() or "timeout" in str(err).lower():
            LOGGER.error("Ошибка соединения с Mistral API: %s", err)
            raise ConfigEntryNotReady(f"Ошибка соединения с Mistral API: {err}") from err
        LOGGER.error("Ошибка инициализации клиента Mistral: %s", err)
        raise ConfigEntryError(f"Ошибка инициализации клиента Mistral: {err}") from err
    else:
        entry.runtime_data = client
        LOGGER.debug("Клиент Mistral успешно инициализирован, настройка платформ: %s", PLATFORMS)
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
        return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Выгрузка интеграции Mistral AI Conversation."""
    LOGGER.debug("Выгрузка записи конфигурации для Mistral AI Conversation")
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        LOGGER.error("Ошибка выгрузки платформ: %s", PLATFORMS)
        return False
    return True