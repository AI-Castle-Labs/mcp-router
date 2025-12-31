# No API Keys Required!

## How It Works

The MCP Router **does NOT need API keys** - it only recommends which model Cursor should use. Cursor handles all API calls with its own configured keys.

```
Your Query → Router Analyzes → Recommends Model → Cursor Uses Model (with its own API keys)
```

## Benefits

✅ **No API key management** - Cursor already has your keys configured
✅ **Simpler setup** - Just install MCP SDK and configure Cursor
✅ **Secure** - API keys stay in Cursor, not in the router
✅ **Flexible** - Router can recommend models without needing access to them

## Setup (No API Keys Needed)

1. **Install MCP SDK:**
   ```bash
   pip3 install --user mcp
   ```

2. **Run setup script:**
   ```bash
   ./scripts/setup_cursor.sh
   ```
   (No API keys needed in the config!)

3. **Restart Cursor** - That's it!

## How Cursor Uses It

When you type a query in Cursor:

1. **Router analyzes** your query (task type, complexity, etc.)
2. **Router recommends** the best model (e.g., "Use GPT-4o-mini for this code task")
3. **Cursor receives** the recommendation
4. **Cursor makes API call** using the recommended model with its own API keys
5. **You get response** from the optimally selected model

## MCP Tools Available

- `route_query` - Get routing decision with reasoning
- `get_model_recommendation` - Get model ID for Cursor to use
- `list_models` - See all available models
- `analyze_query` - Analyze query characteristics
- `get_routing_stats` - View routing statistics

## Example Flow

**You:** "Write a Python function to sort a list"

**Router:** 
```json
{
  "recommended_model": {
    "model_id": "gpt-4o-mini",
    "provider": "openai",
    "name": "GPT-4o-mini"
  },
  "confidence": 0.92,
  "reasoning": "Model is optimized for code_generation tasks; Selected for low latency"
}
```

**Cursor:** Uses `gpt-4o-mini` with its own OpenAI API key → Returns response

## Configuration

The MCP config doesn't need API keys:

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

That's it! No API keys needed because Cursor handles all the API calls.



