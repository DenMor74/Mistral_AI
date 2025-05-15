"""Conversation platform for Mistral AI integration."""
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.components.conversation import ConversationAgent
from .conversation import MistralConversationEntity

async def async_get_agent(hass: HomeAssistant, config_entry: ConfigEntry) -> ConversationAgent:
    """Get the conversation agent for Mistral AI."""
    entity = hass.data.get(DOMAIN, {}).get(config_entry.entry_id)
    if entity is None:
        _LOGGER.error("MistralConversationEntity not found for config entry %s", config_entry.entry_id)
        raise ValueError("MistralConversationEntity not initialized")
    _LOGGER.debug("Returning conversation agent: %s", entity.entity_id)
    return entity