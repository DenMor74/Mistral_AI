
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from . import async_get_conversation_agent

async def async_get_agent(hass: HomeAssistant, config_entry: ConfigEntry):
    return await async_get_conversation_agent(hass, config_entry)
