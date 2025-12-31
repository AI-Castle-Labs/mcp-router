#!/usr/bin/env python3
"""
MCP Router Client - Execute queries using routed models
"""

import os
import json
from typing import Optional, Dict, Any, Iterator

try:
    from .router import MCPRouter, RoutingDecision, QueryContext
except ImportError:
    from router import MCPRouter, RoutingDecision, QueryContext

try:
    from openai import OpenAI as OpenAIClient
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class MCPRouterClient:
    """Client for executing queries through the router."""
    
    def __init__(self, router: Optional[MCPRouter] = None):
        """Initialize the client."""
        self.router = router or MCPRouter()
        self.openai_clients = {}
        self.anthropic_clients = {}
    
    def _get_openai_client(self, api_key: Optional[str] = None) -> Optional[Any]:
        """Get or create OpenAI client."""
        if not HAS_OPENAI:
            return None
        
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        
        if key not in self.openai_clients:
            self.openai_clients[key] = OpenAIClient(api_key=key)
        
        return self.openai_clients[key]
    
    def _get_anthropic_client(self, api_key: Optional[str] = None) -> Optional[Any]:
        """Get or create Anthropic client."""
        if not HAS_ANTHROPIC:
            return None
        
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        
        if key not in self.anthropic_clients:
            self.anthropic_clients[key] = anthropic.Anthropic(api_key=key)
        
        return self.anthropic_clients[key]
    
    def execute(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        strategy: str = "balanced",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a query using the routed model.
        
        Args:
            query: User query
            system_prompt: Optional system prompt
            strategy: Routing strategy
            **kwargs: Additional parameters for the model
        
        Returns:
            Response dictionary with content, model info, and metadata
        """
        # Route query
        decision = self.router.route(query, strategy=strategy)
        model = decision.selected_model
        
        # Execute based on provider
        if model.provider == "openai":
            return self._execute_openai(model, query, system_prompt, **kwargs)
        elif model.provider == "anthropic":
            return self._execute_anthropic(model, query, system_prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {model.provider}")
    
    def _execute_openai(
        self,
        model: Any,
        query: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute query using OpenAI."""
        client = self._get_openai_client()
        if not client:
            raise ValueError("OpenAI API key not found")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        
        response = client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            **kwargs
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": model.model_id,
            "provider": model.provider,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "metadata": {
                "finish_reason": response.choices[0].finish_reason
            }
        }
    
    def _execute_anthropic(
        self,
        model: Any,
        query: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute query using Anthropic."""
        client = self._get_anthropic_client()
        if not client:
            raise ValueError("Anthropic API key not found")
        
        messages = [{"role": "user", "content": query}]
        
        response = client.messages.create(
            model=model.model_id,
            messages=messages,
            system=system_prompt or "",
            **kwargs
        )
        
        return {
            "content": response.content[0].text,
            "model": model.model_id,
            "provider": model.provider,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            "metadata": {}
        }
    
    def stream(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        strategy: str = "balanced",
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream response from routed model.
        
        Yields:
            Chunks of the response
        """
        decision = self.router.route(query, strategy=strategy)
        model = decision.selected_model
        
        if not model.supports_streaming:
            # Fallback to non-streaming
            result = self.execute(query, system_prompt, strategy, **kwargs)
            yield result
            return
        
        if model.provider == "openai":
            yield from self._stream_openai(model, query, system_prompt, **kwargs)
        elif model.provider == "anthropic":
            yield from self._stream_anthropic(model, query, system_prompt, **kwargs)
    
    def _stream_openai(
        self,
        model: Any,
        query: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Stream response from OpenAI."""
        client = self._get_openai_client()
        if not client:
            raise ValueError("OpenAI API key not found")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        
        stream = client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {
                    "content": chunk.choices[0].delta.content,
                    "model": model.model_id,
                    "done": False
                }
        
        yield {"done": True, "model": model.model_id}
    
    def _stream_anthropic(
        self,
        model: Any,
        query: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Stream response from Anthropic."""
        client = self._get_anthropic_client()
        if not client:
            raise ValueError("Anthropic API key not found")
        
        messages = [{"role": "user", "content": query}]
        
        with client.messages.stream(
            model=model.model_id,
            messages=messages,
            system=system_prompt or "",
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield {
                    "content": text,
                    "model": model.model_id,
                    "done": False
                }
        
        yield {"done": True, "model": model.model_id}

