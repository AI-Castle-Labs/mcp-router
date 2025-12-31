#!/usr/bin/env python3
"""
Cursor Wrapper - Simplified interface for Cursor IDE integration
This wrapper makes it easy for Cursor to use the router for model selection.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router import MCPRouter
from src.client import MCPRouterClient


class CursorRouterWrapper:
    """
    Simplified wrapper for Cursor IDE integration.
    Provides a clean interface for Cursor to route queries and get model recommendations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the wrapper."""
        self.router = MCPRouter(config_path)
        self.client = MCPRouterClient(self.router)
        self.default_strategy = os.getenv("ROUTING_STRATEGY", "balanced")
    
    def get_best_model(
        self,
        query: str,
        strategy: Optional[str] = None,
        return_full_decision: bool = False
    ) -> Dict[str, Any]:
        """
        Get the best model for a query.
        
        Args:
            query: The user query/prompt
            strategy: Routing strategy (balanced, cost, speed, quality)
            return_full_decision: If True, return full routing decision
        
        Returns:
            Dictionary with model information or full routing decision
        """
        strategy = strategy or self.default_strategy
        decision = self.router.route(query, strategy=strategy)
        
        if return_full_decision:
            return {
                "model_id": decision.selected_model.model_id,
                "model_name": decision.selected_model.name,
                "provider": decision.selected_model.provider,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "estimated_cost": decision.estimated_cost,
                "estimated_latency_ms": decision.estimated_latency_ms,
                "alternatives": [
                    {
                        "model_id": alt.model_id,
                        "model_name": alt.name,
                        "provider": alt.provider
                    }
                    for alt in decision.alternatives[:3]
                ]
            }
        else:
            return {
                "model_id": decision.selected_model.model_id,
                "model_name": decision.selected_model.name,
                "provider": decision.selected_model.provider,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            }
    
    def execute_with_routing(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route and execute a query.
        
        Args:
            query: User query
            system_prompt: Optional system prompt
            strategy: Routing strategy
        
        Returns:
            Response with content and model information
        """
        strategy = strategy or self.default_strategy
        
        # Route first to get decision
        decision = self.router.route(query, strategy=strategy)
        
        # Execute
        result = self.client.execute(
            query,
            system_prompt=system_prompt,
            strategy=strategy
        )
        
        return {
            "content": result["content"],
            "model_used": {
                "model_id": result["model"],
                "provider": result["provider"]
            },
            "routing": {
                "selected_model": decision.selected_model.name,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            },
            "usage": result["usage"]
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to understand its characteristics.
        
        Args:
            query: The query to analyze
        
        Returns:
            Analysis results
        """
        context = self.router.analyzer.analyze(query)
        
        return {
            "query": query,
            "task_type": context.task_type.value if context.task_type else None,
            "complexity": context.complexity.value if context.complexity else None,
            "estimated_tokens": context.estimated_tokens,
            "requires_streaming": context.requires_streaming,
            "requires_multimodal": context.requires_multimodal,
            "requires_embeddings": context.requires_embeddings
        }
    
    def list_available_models(self) -> list[Dict[str, Any]]:
        """List all available models."""
        return [
            {
                "name": model.name,
                "model_id": model.model_id,
                "provider": model.provider,
                "context_window": model.context_window,
                "cost_per_1k_input": model.cost_per_1k_tokens_input,
                "cost_per_1k_output": model.cost_per_1k_tokens_output,
                "avg_latency_ms": model.avg_latency_ms,
                "reasoning_quality": model.reasoning_quality,
                "code_quality": model.code_quality
            }
            for model in self.router.models.values()
        ]


def main():
    """CLI interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cursor Router Wrapper")
    parser.add_argument("query", help="Query to route")
    parser.add_argument("--strategy", choices=["balanced", "cost", "speed", "quality"],
                       default="balanced", help="Routing strategy")
    parser.add_argument("--execute", action="store_true", help="Execute query")
    parser.add_argument("--analyze", action="store_true", help="Analyze query only")
    
    args = parser.parse_args()
    
    wrapper = CursorRouterWrapper()
    
    if args.analyze:
        analysis = wrapper.analyze_query(args.query)
        print(json.dumps(analysis, indent=2))
    elif args.execute:
        result = wrapper.execute_with_routing(args.query, strategy=args.strategy)
        print(json.dumps(result, indent=2))
    else:
        decision = wrapper.get_best_model(args.query, strategy=args.strategy, return_full_decision=True)
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()




