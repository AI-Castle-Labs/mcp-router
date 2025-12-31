#!/usr/bin/env python3
"""
Example: Context-Aware Routing with Chat History

This example demonstrates how the mcp-router uses chat history
to make smarter model selection decisions.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from router import MCPRouter


def example_basic_query():
    """Example: Basic query without chat history."""
    print("=" * 70)
    print("Example 1: Basic Query (No Chat History)")
    print("=" * 70)

    router = MCPRouter()

    query = "Fix the authentication bug"
    decision = router.route(query, strategy="balanced")

    print(f"\nQuery: {query}")
    print(f"Selected Model: {decision.selected_model.name}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Estimated Cost: ${decision.estimated_cost:.6f}")
    print()


def example_with_chat_history():
    """Example: Query with chat history for context-aware routing."""
    print("=" * 70)
    print("Example 2: Query WITH Chat History (Context-Aware)")
    print("=" * 70)

    router = MCPRouter()

    # Simulate a chat history showing a complex debugging session
    chat_history = [
        {
            "role": "user",
            "content": "I'm working on auth.py and users can't log in with OAuth",
            "timestamp": int(time.time()) - 300
        },
        {
            "role": "assistant",
            "content": "Let me check the OAuth flow in auth.py. Can you show me the error?",
            "timestamp": int(time.time()) - 295
        },
        {
            "role": "user",
            "content": "Here's the error: 'Token validation failed'. It happens in routes.py line 45",
            "timestamp": int(time.time()) - 290
        },
        {
            "role": "assistant",
            "content": "I see the issue. The token verification is failing. Let me examine token_utils.py",
            "timestamp": int(time.time()) - 285
        },
        {
            "role": "user",
            "content": "We're also using Redis for session storage. Could that be related?",
            "timestamp": int(time.time()) - 280
        },
        {
            "role": "assistant",
            "content": "Good point. Let me check redis_client.py and database.py as well",
            "timestamp": int(time.time()) - 275
        }
    ]

    query = "Fix the authentication bug"
    decision = router.route(query, strategy="balanced", chat_history=chat_history)

    print(f"\nQuery: {query}")
    print(f"\nChat History: {len(chat_history)} messages")
    print(f"Selected Model: {decision.selected_model.name}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Estimated Cost: ${decision.estimated_cost:.6f}")
    print()


def example_deep_context():
    """Example: Deep context with many files and complex conversation."""
    print("=" * 70)
    print("Example 3: Deep Context (Multi-file Refactoring)")
    print("=" * 70)

    router = MCPRouter()

    # Simulate a long conversation about refactoring
    long_content = """
    We need to refactor the entire authentication system across multiple files.
    The current implementation in auth.py, routes.py, middleware.py, and
    token_utils.py is tightly coupled and hard to test. We should separate
    concerns and implement proper dependency injection.

    Files involved:
    - auth.py (main authentication logic)
    - routes.py (API endpoints)
    - middleware.py (auth middleware)
    - token_utils.py (JWT handling)
    - database.py (user model)
    - redis_client.py (session storage)
    - config.py (auth configuration)
    - tests/test_auth.py (authentication tests)

    The refactoring needs to:
    1. Extract interface for authentication providers
    2. Implement OAuth2, JWT, and API key auth separately
    3. Add comprehensive error handling
    4. Improve test coverage
    5. Add proper logging
    6. Update documentation
    """

    chat_history = [
        {
            "role": "user",
            "content": long_content,
            "timestamp": int(time.time()) - 600
        },
        {
            "role": "assistant",
            "content": "This is a complex refactoring. Let me analyze the current architecture...",
            "timestamp": int(time.time()) - 595
        },
        {
            "role": "user",
            "content": "We're using Python with FastAPI, PostgreSQL, and Redis",
            "timestamp": int(time.time()) - 590
        }
    ]

    query = "Start refactoring the authentication system with the strategy pattern"
    decision = router.route(query, strategy="quality", chat_history=chat_history)

    print(f"\nQuery: {query}")
    print(f"\nChat History: {len(chat_history)} messages")
    print(f"Context Depth: Deep (multi-file, complex refactoring)")
    print(f"Selected Model: {decision.selected_model.name}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Estimated Cost: ${decision.estimated_cost:.6f}")

    # Show context window advantage
    print(f"\nModel Context Window: {decision.selected_model.context_window:,} tokens")
    print(f"This is important for handling large codebases!")
    print()


def example_polyglot_project():
    """Example: Polyglot project with multiple languages."""
    print("=" * 70)
    print("Example 4: Polyglot Project (Python + JavaScript + Rust)")
    print("=" * 70)

    router = MCPRouter()

    chat_history = [
        {
            "role": "user",
            "content": "Need to update the API client in client.ts to match our new Python backend",
            "timestamp": int(time.time()) - 400
        },
        {
            "role": "assistant",
            "content": "I'll update the TypeScript client to match your FastAPI backend",
            "timestamp": int(time.time()) - 395
        },
        {
            "role": "user",
            "content": "Also need to update the Rust service that processes the data",
            "timestamp": int(time.time()) - 390
        },
        {
            "role": "assistant",
            "content": "Working on data_processor.rs to match the new schema",
            "timestamp": int(time.time()) - 385
        }
    ]

    query = "Make sure all three services are consistent"
    decision = router.route(query, strategy="quality", chat_history=chat_history)

    print(f"\nQuery: {query}")
    print(f"\nDetected Languages: Python, JavaScript/TypeScript, Rust")
    print(f"Selected Model: {decision.selected_model.name}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Code Quality Score: {decision.selected_model.code_quality:.2f}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "Context-Aware Model Routing with Chat History".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Run examples
    example_basic_query()
    print("\n")

    example_with_chat_history()
    print("\n")

    example_deep_context()
    print("\n")

    example_polyglot_project()

    print("=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Chat history provides context about the task complexity")
    print("2. Files mentioned influence model selection (multi-file = higher quality)")
    print("3. Error patterns trigger debugging-optimized models")
    print("4. Programming languages detected guide code-focused models")
    print("5. Deep conversations favor models with larger context windows")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
