"""Mistral AI conversation integration for Home Assistant."""
import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Mistral AI conversation from a config entry."""
    try:
        # Forward setup to the conversation platform
        await hass.config_entries.async_forward_entry_setup(entry, "conversation")
        return True
    except Exception as err:
        _LOGGER.error("Failed to set up Mistral AI conversation: %s", err)
        raise ConfigEntryNotReady(f"Failed to set up Mistral AI conversation: {err}") from err

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_forward_entry_unload(entry, "conversation")