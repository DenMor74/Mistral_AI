"""Constants for the Mistral AI conversation integration."""
DOMAIN = "mistral_conversation"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_MAX_TOKENS = "max_tokens"
CONF_CHAT_HISTORY = "chat_history"
CONF_PROMPT = "prompt"

DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_CHAT_HISTORY = 10
DEFAULT_PROMPT = "You are a helpful AI assistant integrated into Home Assistant, providing accurate and concise answers."