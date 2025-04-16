# src/agents/models/openrouter_provider.py
from __future__ import annotations

from openai import AsyncOpenAI

from .interface import Model, ModelProvider
from .openai_provider import OpenAIProvider
from .openai_chatcompletions import OpenAIChatCompletionsModel
from ..version import __version__

DEFAULT_MODEL: str = "openai/gpt-4o"

class OpenRouterProvider(OpenAIProvider):
    """
    Implementation of ModelProvider that uses the OpenRouter API.
    
    OpenRouter provides access to multiple AI model providers through a single API
    compatible with the OpenAI interface.
    """
    
    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_referer: str | None = None,
        site_title: str | None = None,
        use_responses: bool | None = None,  # Kept for compatibility, but ignored
    ) -> None:
        """Create a new OpenRouter provider.

        Args:
            api_key: The API key for OpenRouter.
            http_referer: Optional site URL for rankings on openrouter.ai.
            site_title: Optional site title for rankings on openrouter.ai.
            use_responses: Ignored. OpenRouter only supports the Chat Completions API format.
        """
        # Initialize the OpenAI provider with the OpenRouter base URL
        # Force use_responses to False since OpenRouter only supports Chat Completions
        super().__init__(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            use_responses=False,
        )
        
        # Store custom headers
        self._http_referer = http_referer
        self._site_title = site_title
        
    def _get_client(self) -> AsyncOpenAI:
        """Get or create an AsyncOpenAI client configured for OpenRouter."""
        # Get the client from the parent class
        client = super()._get_client()
        
        # Add OpenRouter specific headers if they haven't been added yet
        if self._http_referer and "HTTP-Referer" not in client.default_headers:
            client.default_headers["HTTP-Referer"] = self._http_referer
            
        if self._site_title and "X-Title" not in client.default_headers:
            client.default_headers["X-Title"] = self._site_title
            
        return client
        
    def get_model(self, model_name: str | None) -> Model:
        """
        Get an implementation of Model using the specified model name.
        
        Args:
            model_name: The name of the model to use, with provider prefix 
                (e.g., "openai/gpt-4o", "anthropic/claude-3-opus").
                If None, uses the default model.
        
        Returns:
            Model: An implementation of the Model interface (always OpenAIChatCompletionsModel).
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
            
        # Always use the Chat Completions implementation since OpenRouter
        # doesn't fully support the Responses API format
        client = self._get_client()
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)