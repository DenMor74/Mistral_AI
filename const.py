"""Константы для интеграции Mistral AI Conversation."""
import logging

DOMAIN = "mistral_conversation"
LOGGER = logging.getLogger(__name__)

CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_MAX_TOKENS = "max_tokens"
CONF_CHAT_HISTORY = "chat_history"
CONF_PROMPT = "prompt"
CONF_LLM_HASS_API = "llm_hass_api"

DEFAULT_MODEL = "mistral-large-latest"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CHAT_HISTORY = 10
DEFAULT_PROMPT = "Ты полезный ИИ-ассистент, интегрированный в Home Assistant. Отвечай кратко и по делу, используя предоставленные инструменты, если требуется."
DOMAIN = "mistral_conversation"