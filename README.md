# 🚀 MCP Router

> **Intelligent Model Context Protocol Router for Cursor IDE**
> 
> Automatically selects the optimal LLM model for each task based on query analysis, complexity, and your preferred strategy.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CURSOR IDE                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         User Query                                      │ │
│  │   "Refactor this authentication system across multiple files"          │ │
│  └──────────────────────────────┬─────────────────────────────────────────┘ │
│                                 │                                            │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      MCP Router Server                                  │ │
│  │  ┌──────────────────┐    ┌───────────────────┐   ┌──────────────────┐  │ │
│  │  │  Query Analyzer  │───▶│   Model Scorer    │──▶│ Routing Decision │  │ │
│  │  │                  │    │                   │   │                  │  │ │
│  │  │ • Task Type      │    │ • Quality Score   │   │ • Selected Model │  │ │
│  │  │ • Complexity     │    │ • Cost Score      │   │ • Confidence     │  │ │
│  │  │ • Requirements   │    │ • Speed Score     │   │ • Reasoning      │  │ │
│  │  │ • Token Estimate │    │ • Strategy Weight │   │ • Alternatives   │  │ │
│  │  └──────────────────┘    └───────────────────┘   └──────────────────┘  │ │
│  └──────────────────────────────┬─────────────────────────────────────────┘ │
│                                 │                                            │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     Model Registry (17 Models)                          │ │
│  │                                                                         │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │ │
│  │  │  FLAGSHIP   │ │  REASONING  │ │ NATIVE/FAST │ │  BUDGET/LEGACY  │   │ │
│  │  │             │ │             │ │             │ │                 │   │ │
│  │  │ • GPT-5.2   │ │ • o3        │ │ • Composer1 │ │ • GPT-4o-mini   │   │ │
│  │  │ • Claude4.5 │ │ • o3-mini   │ │ • Gemini 3  │ │ • Claude Haiku  │   │ │
│  │  │   Opus     │ │ • Claude3.7 │ │   Pro/Flash │ │ • DeepSeek V3   │   │ │
│  │  │ • Claude4.5 │ │   Sonnet   │ │             │ │ • DeepSeek R1   │   │ │
│  │  │   Sonnet   │ │             │ │             │ │                 │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘   │ │
│  └──────────────────────────────┬─────────────────────────────────────────┘ │
│                                 │                                            │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Cursor Executes Query                                │ │
│  │            (Using its own API keys for selected model)                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────┐      ┌──────────────┐      ┌───────────────┐      ┌────────────┐
│  Query   │─────▶│   Analyze    │─────▶│     Score     │─────▶│  Recommend │
└──────────┘      └──────────────┘      └───────────────┘      └────────────┘
                         │                      │                      │
                         ▼                      ▼                      ▼
                  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
                  │ Task Type:  │       │ Apply       │       │ Model:      │
                  │ • reasoning │       │ Strategy:   │       │ Claude 4.5  │
                  │ • code_gen  │       │ • balanced  │       │ Sonnet      │
                  │ • edit      │       │ • quality   │       │             │
                  │ Complexity: │       │ • speed     │       │ Confidence: │
                  │ • medium    │       │ • cost      │       │ 88.45%      │
                  └─────────────┘       └─────────────┘       └─────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Intelligent Routing** | Automatically selects the best model based on query analysis |
| 🧠 **Context-Aware Routing** | Uses chat history and conversation context for smarter model selection |
| 📊 **4 Routing Strategies** | `balanced` / `cost` / `speed` / `quality` |
| 🔍 **Query Analysis** | Detects task type, complexity, and special requirements |
| 💬 **Chat History Analysis** | Analyzes conversation patterns, topics, files, languages, and complexity |
| 💰 **Cost Estimation** | Estimates costs before execution |
| ⚡ **17 Models** | Latest 2025 models from OpenAI, Anthropic, Google, Cursor, DeepSeek |
| 🔧 **Cursor Native** | Zero API keys needed - Cursor handles execution |

---

## 🏆 Supported Models (2025)

### Tier 1: Flagship Models (Complex Architecture & Refactoring)

| Model | Provider | Context | Cost (in/out) | Quality |
|-------|----------|---------|---------------|---------|
| **GPT-5.2** | OpenAI | 256K | $5.00/$15.00 | 0.99/0.98 |
| **Claude 4.5 Opus** | Anthropic | 200K | $25.00/$75.00 | 0.99/0.99 |
| **Claude 4.5 Sonnet** | Anthropic | 200K | $5.00/$25.00 | 0.97/0.98 |

### Tier 2: Reasoning Models (Chain of Thought)

| Model | Provider | Context | Cost (in/out) | Quality |
|-------|----------|---------|---------------|---------|
| **o3** | OpenAI | 200K | $10.00/$40.00 | 0.99/0.95 |
| **o3-mini (High)** | OpenAI | 128K | $1.50/$6.00 | 0.95/0.92 |
| **Claude 3.7 Sonnet** | Anthropic | 200K | $4.00/$20.00 | 0.96/0.96 |

### Tier 3: Native & Fast Models

| Model | Provider | Context | Cost (in/out) | Quality |
|-------|----------|---------|---------------|---------|
| **Composer 1** | Cursor | 128K | $0.10/$0.30 | 0.88/0.92 |
| **Gemini 3 Pro** | Google | **2M** | $2.00/$8.00 | 0.96/0.94 |
| **Gemini 3 Flash** | Google | 1M | $0.10/$0.40 | 0.88/0.90 |

### Tier 4: Budget/Legacy Models

| Model | Provider | Context | Quality |
|-------|----------|---------|---------|
| GPT-4o / GPT-4o-mini | OpenAI | 128K | 0.95/0.85 |
| Claude 3.5 Sonnet/Haiku | Anthropic | 200K | 0.96/0.88 |
| Gemini 2.0 Pro/Flash | Google | 2M/1M | 0.94/0.85 |
| **DeepSeek V3** | DeepSeek | 128K | 0.92/0.94 |
| **DeepSeek R1** | DeepSeek | 128K | 0.96/0.92 |

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/AI-Castle-Labs/mcp-router.git
cd mcp-router
pip install -r requirements.txt
pip install mcp  # MCP SDK for Cursor integration
```

### 2. Configure Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "version": "1.0",
  "mcpServers": {
    "mcp-router": {
      "command": "python3",
      "args": ["/path/to/mcp-router/src/mcp_server.py"],
      "env": {}
    }
  }
}
```

> **Note:** No API keys needed! Cursor handles all API calls with its own keys.

### 3. Restart Cursor

The MCP router will appear in your agent tools. Use it with:
- `@mcp-router get_model_recommendation "your task description"`
- `@mcp-router analyze_query "your query"`
- `@mcp-router list_models`

---

## 💻 CLI Usage

```bash
# Route a query (shows which model would be selected)
python main.py route "Explain how neural networks work"

# Route with strategy
python main.py route "Refactor this codebase" --strategy quality

# List all registered models
python main.py list

# Show routing statistics
python main.py stats
```

### Example Output

```
============================================================
Routing Decision
============================================================
Query: Refactor this complex authentication system...

Selected Model: Claude 4.5 Sonnet
Model ID: claude-4.5-sonnet
Provider: anthropic
Confidence: 88.45%

Reasoning: Model is optimized for code_edit tasks; Selected for highest quality

Alternatives:
  - Composer 1 (composer-1)
  - Claude 3.5 Haiku (claude-3-5-haiku-20241022)
  - GPT-4o-mini (gpt-4o-mini)
```

---

## 🎯 Routing Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `balanced` | Optimizes for cost, speed, and quality equally | General use |
| `quality` | Prioritizes highest capability models | Complex tasks, refactoring |
| `speed` | Prioritizes fastest response time | Quick edits, simple tasks |
| `cost` | Prioritizes cheapest models | Budget-conscious usage |

---

## 🐍 Python API

```python
from src.router import MCPRouter

# Initialize router (loads 17 default models)
router = MCPRouter()

# Route a query
decision = router.route(
    "Analyze this codebase architecture",
    strategy="quality"
)

print(f"Selected: {decision.selected_model.name}")
print(f"Model ID: {decision.selected_model.model_id}")
print(f"Confidence: {decision.confidence:.1%}")
print(f"Reasoning: {decision.reasoning}")

# Get alternatives
for alt in decision.alternatives[:3]:
    print(f"  Alternative: {alt.name}")
```

---

## 📁 Project Structure

```
mcp-router/
├── src/
│   ├── router.py          # Core routing logic + 17 model definitions
│   ├── mcp_server.py       # MCP server for Cursor integration
│   ├── client.py           # API client for model execution
│   └── cursor_wrapper.py   # Cursor-specific utilities
├── config/
│   └── cursor_mcp_config.json  # Template for Cursor config
├── scripts/
│   └── setup_cursor.sh     # Automated setup script
├── docs/
│   ├── cursor_integration.md
│   ├── QUICKSTART_CURSOR.md
│   └── AGENT_SETTINGS.md
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🔧 Adding Custom Models

```python
from src.router import MCPRouter, ModelCapabilities, TaskType

router = MCPRouter()

router.register_model(ModelCapabilities(
    name="My Custom Model",
    provider="custom",
    model_id="custom-model-v1",
    supports_reasoning=True,
    supports_code=True,
    supports_streaming=True,
    max_tokens=8192,
    context_window=32000,
    cost_per_1k_tokens_input=1.0,
    cost_per_1k_tokens_output=2.0,
    avg_latency_ms=600,
    reasoning_quality=0.85,
    code_quality=0.90,
    speed_score=0.80,
    preferred_tasks=[TaskType.CODE_GENERATION],
    api_key_env_var="CUSTOM_API_KEY"
))
```

---

## 🎮 Cursor Commands

Create `.cursor/commands/route.md`:

```markdown
---
description: "Get model recommendation from MCP router for the current task"
---

Use the MCP router to determine the best model for the task at hand.

1. Analyze the current context
2. Call `@mcp-router get_model_recommendation` with task description
3. Present the recommendation with confidence and alternatives
4. Suggest switching models if needed
```

---

## 📊 MCP Tools Available

| Tool | Description |
|------|-------------|
| `route_query` | Route a query and get model recommendation (supports chat_history) |
| `get_model_recommendation` | Get recommendation without execution (supports chat_history) |
| `analyze_chat_summary` | Analyze chat history text to extract routing signals |
| `list_models` | List all 17 registered models |
| `get_routing_stats` | Get usage statistics |
| `analyze_query` | Analyze query characteristics |

### Context-Aware Routing with Chat History

The router can now analyze chat history to make smarter routing decisions:

```javascript
// Example: Using chat history for context-aware routing
{
  "query": "Fix the authentication bug we discussed",
  "strategy": "quality",
  "chat_history": [
    {
      "role": "user",
      "content": "I'm working on auth.py and users can't log in",
      "timestamp": 1704067200
    },
    {
      "role": "assistant",
      "content": "Let me check the authentication flow...",
      "timestamp": 1704067205
    }
  ]
}
```

The router analyzes chat history to detect:
- **Context depth**: Shallow/medium/deep based on token count
- **Dominant task type**: Code generation, editing, debugging, etc.
- **Programming languages**: Detects Python, JavaScript, Rust, etc.
- **Files mentioned**: Tracks files being worked on
- **Error patterns**: Identifies debugging sessions
- **Topics**: Authentication, database, API, testing, etc.
- **Complexity**: Based on files, languages, and conversation depth

These signals influence model selection:
- Deep context → Models with larger context windows
- Debugging sessions → High-reasoning models
- Multi-file tasks → Code-focused models
- Multiple languages → Polyglot-capable models

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built for the Cursor IDE ecosystem</b><br>
  <a href="https://github.com/AI-Castle-Labs/mcp-router">AI Castle Labs</a>
</p>
