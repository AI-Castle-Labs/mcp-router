#!/usr/bin/env python3
"""
MCP Router CLI - Command-line interface for the router
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.router import MCPRouter, TaskType, Complexity
from src.client import MCPRouterClient


def main():
    parser = argparse.ArgumentParser(
        description="MCP Router - Intelligent Model Routing System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Route command
    route_parser = subparsers.add_parser("route", help="Route a query to the best model")
    route_parser.add_argument("query", help="Query to route")
    route_parser.add_argument(
        "--strategy",
        choices=["balanced", "cost", "speed", "quality"],
        default="balanced",
        help="Routing strategy"
    )
    route_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the query using the routed model"
    )
    route_parser.add_argument(
        "--system-prompt",
        help="System prompt for execution"
    )
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List registered models")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show routing statistics")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "--save",
        help="Save current configuration to file"
    )
    config_parser.add_argument(
        "--load",
        help="Load configuration from file"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    router = MCPRouter()
    
    if args.command == "route":
        # Route query
        decision = router.route(args.query, strategy=args.strategy)
        
        print(f"\n{'='*60}")
        print(f"Routing Decision")
        print(f"{'='*60}")
        print(f"Query: {args.query[:100]}...")
        print(f"\nSelected Model: {decision.selected_model.name}")
        print(f"Model ID: {decision.selected_model.model_id}")
        print(f"Provider: {decision.selected_model.provider}")
        print(f"Confidence: {decision.confidence:.2%}")
        print(f"\nReasoning: {decision.reasoning}")
        print(f"\nEstimated Cost: ${decision.estimated_cost:.6f}")
        print(f"Estimated Latency: {decision.estimated_latency_ms}ms")
        
        if decision.alternatives:
            print(f"\nAlternatives:")
            for alt in decision.alternatives[:3]:
                print(f"  - {alt.name} ({alt.model_id})")
        
        # Execute if requested
        if args.execute:
            print(f"\n{'='*60}")
            print(f"Executing Query...")
            print(f"{'='*60}\n")
            
            try:
                client = MCPRouterClient(router)
                result = client.execute(
                    args.query,
                    system_prompt=args.system_prompt,
                    strategy=args.strategy
                )
                
                print(f"\nResponse from {result['model']}:\n")
                print(result['content'])
                print(f"\n\nUsage: {result['usage']}")
            except Exception as e:
                print(f"Error executing query: {e}")
                return 1
    
    elif args.command == "list":
        print(f"\n{'='*60}")
        print(f"Registered Models")
        print(f"{'='*60}\n")
        
        for model in router.models.values():
            print(f"Name: {model.name}")
            print(f"  ID: {model.model_id}")
            print(f"  Provider: {model.provider}")
            print(f"  Context Window: {model.context_window:,} tokens")
            print(f"  Cost: ${model.cost_per_1k_tokens_input:.2f}/1k input, ${model.cost_per_1k_tokens_output:.2f}/1k output")
            print(f"  Latency: {model.avg_latency_ms}ms")
            print(f"  Quality: Reasoning={model.reasoning_quality:.2f}, Code={model.code_quality:.2f}")
            print()
    
    elif args.command == "stats":
        stats = router.get_routing_stats()
        print(f"\n{'='*60}")
        print(f"Routing Statistics")
        print(f"{'='*60}\n")
        print(f"Total Routes: {stats.get('total_routes', 0)}")
        
        if stats.get('model_usage'):
            print(f"\nModel Usage:")
            for model_id, count in stats['model_usage'].items():
                print(f"  {model_id}: {count}")
        
        if stats.get('avg_confidence'):
            print(f"\nAverage Confidence: {stats['avg_confidence']:.2%}")
    
    elif args.command == "config":
        if args.save:
            router.save_config(args.save)
            print(f"Configuration saved to {args.save}")
        elif args.load:
            router.load_config(args.load)
            print(f"Configuration loaded from {args.load}")
        else:
            print("Use --save or --load to manage configuration")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

