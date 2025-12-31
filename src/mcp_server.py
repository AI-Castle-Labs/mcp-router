#!/usr/bin/env python3
"""
MCP Server for Cursor IDE Integration
Exposes the router as an MCP server so Cursor can use it for intelligent model routing.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    # Fallback if MCP SDK structure is different
    try:
        from mcp import Server, types
        from mcp.server.stdio import stdio_server
        HAS_MCP = True
    except ImportError:
        HAS_MCP = False
        print("Warning: MCP SDK not installed. Install with: pip install mcp")

from src.router import MCPRouter, QueryContext, TaskType, Complexity, ChatSummary, ChatSummaryAnalyzer
# Note: We don't need MCPRouterClient since Cursor will handle API calls


class CursorMCPRouter:
    """MCP Server wrapper for the router."""
    
    # Paths to search for chat history/context
    LEDGER_PATHS = [
        ".cursor/chat_history/ledger.md",
        ".cursor/ledger.md",
        "ledger.md",
        "docs/chat_history/ledger.md",
    ]
    
    # Cursor's native chat history location (macOS)
    CURSOR_HISTORY_PATHS = [
        Path.home() / "Library/Application Support/Cursor/User/History",
        Path.home() / ".config/Cursor/User/History",  # Linux
        Path(os.environ.get("APPDATA", "")) / "Cursor/User/History",  # Windows
    ]
    
    def __init__(self):
        """Initialize the MCP router server."""
        if not HAS_MCP:
            raise ImportError(
                "MCP SDK required. Install with: pip install mcp"
            )
        
        # Router doesn't need API keys - it just recommends models
        self.router = MCPRouter()
        self.chat_analyzer = ChatSummaryAnalyzer()
        
        # Try to detect workspace root from environment or current directory
        try:
            workspace_env = os.environ.get("CURSOR_WORKSPACE")
            if workspace_env:
                self.workspace_root = Path(workspace_env)
            else:
                self.workspace_root = Path.cwd()
        except Exception:
            # Fallback to current directory if path resolution fails
            self.workspace_root = Path.cwd()
        
        # Initialize server
        try:
            self.server = Server("mcp-router")
        except:
            # Fallback initialization
            self.server = Server()
        
        # Register handlers
        self._register_handlers()
    
    def _find_ledger(self) -> Optional[str]:
        """Find and read the ledger.md file from common locations."""
        for ledger_path in self.LEDGER_PATHS:
            full_path = self.workspace_root / ledger_path
            if full_path.exists():
                try:
                    return full_path.read_text()
                except Exception:
                    continue
        return None
    
    def _get_recent_cursor_history(self, max_files: int = 3) -> Optional[str]:
        """
        Read recent Cursor chat history files.
        Returns combined content from recent sessions.
        """
        for history_path in self.CURSOR_HISTORY_PATHS:
            if not history_path.exists():
                continue
            
            try:
                # Find session directories, sorted by modification time
                sessions = sorted(
                    [d for d in history_path.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True
                )[:max_files]
                
                combined_content = []
                for session_dir in sessions:
                    # Look for .md files in each session
                    md_files = sorted(
                        session_dir.glob("*.md"),
                        key=lambda f: f.stat().st_mtime,
                        reverse=True
                    )[:1]  # Most recent .md file per session
                    
                    for md_file in md_files:
                        try:
                            content = md_file.read_text()
                            if len(content) > 100:  # Skip empty/trivial files
                                combined_content.append(content[:5000])  # Limit per file
                        except Exception:
                            continue
                
                if combined_content:
                    return "\n\n---\n\n".join(combined_content)
            except Exception:
                continue
        
        return None
    
    def _get_auto_chat_context(self) -> Optional[ChatSummary]:
        """
        Automatically gather chat context from available sources.
        Priority: 1. ledger.md, 2. Cursor history
        """
        # Try ledger.md first (user-curated summary)
        ledger_content = self._find_ledger()
        if ledger_content:
            return self.chat_analyzer.analyze_from_text(ledger_content)
        
        # Fall back to Cursor's native history
        history_content = self._get_recent_cursor_history()
        if history_content:
            return self.chat_analyzer.analyze_from_text(history_content)
        
        return None
    
    def _register_handlers(self):
        """Register MCP handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="route_query",
                    description="Route a query to the best model based on query characteristics, routing strategy, and optional chat history",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query/prompt to route"
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["balanced", "cost", "speed", "quality"],
                                "description": "Routing strategy",
                                "default": "balanced"
                            },
                            "chat_history": {
                                "type": "array",
                                "description": "Optional chat history as array of message objects with 'role', 'content', 'timestamp' keys",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string"},
                                        "content": {"type": "string"},
                                        "timestamp": {"type": "number"}
                                    }
                                }
                            },
                            "chat_summary": {
                                "type": "string",
                                "description": "Optional text summary of chat history (alternative to chat_history)"
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "Optional system prompt for context"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_model_recommendation",
                    description="Get model recommendation using query analysis and optional chat history/summary for context-aware routing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query/prompt to route"
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["balanced", "cost", "speed", "quality"],
                                "description": "Routing strategy",
                                "default": "balanced"
                            },
                            "chat_history": {
                                "type": "array",
                                "description": "Optional chat history as array of message objects with 'role', 'content', 'timestamp' keys",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string"},
                                        "content": {"type": "string"},
                                        "timestamp": {"type": "number"}
                                    }
                                }
                            },
                            "chat_summary": {
                                "type": "string",
                                "description": "Optional text summary of chat history (alternative to chat_history) to inform model selection"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="analyze_chat_summary",
                    description="Analyze chat history text to extract routing signals (topics, complexity, languages, files)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "summary_text": {
                                "type": "string",
                                "description": "Chat summary text to analyze (e.g., content from ledger.md or conversation summary)"
                            }
                        },
                        "required": ["summary_text"]
                    }
                ),
                Tool(
                    name="list_models",
                    description="List all available models in the router",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_routing_stats",
                    description="Get statistics about routing decisions",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="analyze_query",
                    description="Analyze a query to determine its characteristics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to analyze"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "route_query":
                    query = arguments.get("query", "")
                    strategy = arguments.get("strategy", "balanced")
                    chat_history = arguments.get("chat_history", None)
                    summary_text = arguments.get("chat_summary", None)

                    # Analyze query first to get context
                    context = self.router.analyzer.analyze(query)
                    decision = self.router.route(
                        query,
                        context=context,
                        strategy=strategy,
                        chat_history=chat_history,
                        summary_text=summary_text
                    )
                    
                    result = {
                        "selected_model": {
                            "name": decision.selected_model.name,
                            "model_id": decision.selected_model.model_id,
                            "provider": decision.selected_model.provider
                        },
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "estimated_cost": decision.estimated_cost,
                        "estimated_latency_ms": decision.estimated_latency_ms,
                        "alternatives": [
                            {
                                "name": alt.name,
                                "model_id": alt.model_id,
                                "provider": alt.provider
                            }
                            for alt in decision.alternatives[:3]
                        ]
                    }

                    # Include chat context if available
                    if context.chat_summary:
                        result["chat_context"] = {
                            "total_messages": context.chat_summary.total_messages,
                            "context_depth": context.chat_summary.context_depth,
                            "dominant_task_type": context.chat_summary.dominant_task_type.value if context.chat_summary.dominant_task_type else None,
                            "languages_used": context.chat_summary.languages_used,
                            "files_mentioned_count": len(context.chat_summary.files_mentioned)
                        }
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )]
                
                elif name == "get_model_recommendation":
                    query = arguments.get("query", "")
                    strategy = arguments.get("strategy", "balanced")
                    chat_history = arguments.get("chat_history", None)
                    summary_text = arguments.get("chat_summary", None)
                    
                    # AUTO-DETECT CHAT CONTEXT if not provided
                    auto_chat_summary = None
                    chat_source = None
                    
                    if not chat_history and not summary_text:
                        # Automatically gather chat context from ledger.md or Cursor history
                        auto_chat_summary = self._get_auto_chat_context()
                        if auto_chat_summary:
                            # Determine source for transparency
                            ledger = self._find_ledger()
                            chat_source = "ledger.md" if ledger else "cursor_history"

                    # Analyze query first to get context
                    context = self.router.analyzer.analyze(query)

                    # Route query with chat context (provided or auto-detected)
                    decision = self.router.route(
                        query,
                        context=context,
                        strategy=strategy,
                        chat_history=chat_history,
                        summary_text=summary_text,
                        chat_summary=auto_chat_summary  # Use auto-detected if no manual provided
                    )
                    
                    # Return model recommendation for Cursor to use
                    # Cursor will handle the actual API call with its own keys
                    result = {
                        "recommended_model": {
                            "model_id": decision.selected_model.model_id,
                            "provider": decision.selected_model.provider,
                            "name": decision.selected_model.name
                        },
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "estimated_cost": decision.estimated_cost,
                        "estimated_latency_ms": decision.estimated_latency_ms,
                        "alternatives": [
                            {
                                "model_id": alt.model_id,
                                "provider": alt.provider,
                                "name": alt.name
                            }
                            for alt in decision.alternatives[:3]
                        ],
                        "query_analysis": {
                            "task_type": context.task_type.value if context.task_type else None,
                            "complexity": context.complexity.value if context.complexity else None,
                            "estimated_tokens": context.estimated_tokens
                        }
                    }

                    # Include chat summary analysis if available
                    if context.chat_summary:
                        result["chat_context"] = {
                            "source": chat_source or "provided",  # Where chat context came from
                            "auto_detected": chat_source is not None,  # Was it auto-detected?
                            "total_messages": context.chat_summary.total_messages,
                            "context_depth": context.chat_summary.context_depth,
                            "requires_continuity": context.chat_summary.requires_continuity,
                            "dominant_task_type": context.chat_summary.dominant_task_type.value if context.chat_summary.dominant_task_type else None,
                            "avg_complexity": context.chat_summary.avg_complexity.value if context.chat_summary.avg_complexity else None,
                            "languages_used": context.chat_summary.languages_used,
                            "files_mentioned": context.chat_summary.files_mentioned[:10],
                            "topics": context.chat_summary.topics,
                            "success_rate": context.chat_summary.success_rate
                        }
                    else:
                        result["chat_context"] = {
                            "source": None,
                            "auto_detected": False,
                            "note": "No chat context found. Create .cursor/chat_history/ledger.md or use /summarize command."
                        }
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )]
                
                elif name == "list_models":
                    models = []
                    for model in self.router.models.values():
                        models.append({
                            "name": model.name,
                            "model_id": model.model_id,
                            "provider": model.provider,
                            "context_window": model.context_window,
                            "cost_per_1k_input": model.cost_per_1k_tokens_input,
                            "cost_per_1k_output": model.cost_per_1k_tokens_output,
                            "avg_latency_ms": model.avg_latency_ms,
                            "reasoning_quality": model.reasoning_quality,
                            "code_quality": model.code_quality
                        })
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps({"models": models}, indent=2)
                    )]
                
                elif name == "get_routing_stats":
                    stats = self.router.get_routing_stats()
                    return [TextContent(
                        type="text",
                        text=json.dumps(stats, indent=2)
                    )]
                
                elif name == "analyze_query":
                    query = arguments.get("query", "")
                    context = self.router.analyzer.analyze(query)
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "query": query,
                            "task_type": context.task_type.value if context.task_type else None,
                            "complexity": context.complexity.value if context.complexity else None,
                            "estimated_tokens": context.estimated_tokens,
                            "requires_streaming": context.requires_streaming,
                            "requires_multimodal": context.requires_multimodal,
                            "requires_embeddings": context.requires_embeddings
                        }, indent=2)
                    )]
                
                elif name == "analyze_chat_summary":
                    summary_text = arguments.get("summary_text", "")
                    
                    # Analyze chat summary to extract routing signals
                    chat_analyzer = ChatSummaryAnalyzer()
                    chat_summary = chat_analyzer.analyze_from_text(summary_text)
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "chat_summary": chat_summary.to_dict(),
                            "routing_signals": {
                                "recommended_context_window": "200K+" if chat_summary.context_depth == "deep" else "128K",
                                "needs_high_reasoning": chat_summary.success_rate < 0.8 or chat_summary.avg_complexity in [Complexity.HIGH, Complexity.VERY_HIGH],
                                "needs_code_focus": len(chat_summary.files_mentioned) >= 5 or len(chat_summary.languages_used) >= 2,
                                "continuity_important": chat_summary.requires_continuity,
                                "session_type": "debugging" if chat_summary.success_rate < 0.8 else 
                                               "development" if chat_summary.dominant_task_type in [TaskType.CODE_GENERATION, TaskType.CODE_EDIT] else
                                               "analysis"
                            }
                        }, indent=2)
                    )]
                
                else:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"Unknown tool: {name}"})
                    )]
            
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            init_options = self.server.create_initialization_options()
            await self.server.run(
                read_stream,
                write_stream,
                init_options,
                raise_exceptions=False  # Let server handle exceptions internally
            )


async def main():
    """Main entry point."""
    if not HAS_MCP:
        print("Error: MCP SDK not installed.", file=sys.stderr)
        print("Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)
    
    try:
        router_server = CursorMCPRouter()
        await router_server.run()
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        pass
    except Exception as e:
        import traceback
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

