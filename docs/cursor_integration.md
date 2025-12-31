# Cursor IDE Integration Guide

This guide shows how to integrate the MCP Router with Cursor IDE so it automatically selects the best model for each query.

## Overview

The integration works by:
1. **Query comes in** → Cursor receives your prompt
2. **Router analyzes** → Determines best model based on query characteristics
3. **Model selected** → Router picks optimal model (GPT-4o, Claude, etc.)
4. **Cursor executes** → Uses the selected model for your prompt

## Setup

### 1. Install Dependencies

```bash
cd mcp-router
pip install -r requirements.txt
pip install mcp  # MCP SDK for Cursor integration
```

### 2. Configure Cursor

Add the MCP server to Cursor's configuration:

**On macOS:**
```bash
# Edit Cursor's MCP settings
open ~/Library/Application\ Support/Cursor/User/globalStorage/mcp.json
```

**On Windows:**
```
%APPDATA%\Cursor\User\globalStorage\mcp.json
```

**On Linux:**
```
~/.config/Cursor/User/globalStorage/mcp.json
```

### 3. Add MCP Server Configuration

Add this to your Cursor MCP configuration file:

```json
{
  "version": "1.0",
  "mcpServers": {
    "mcp-router": {
      "command": "python3",
      "args": [
        "/absolute/path/to/mcp-router/src/mcp_server.py"
      ],
      "env": {}
    }
  }
}
```

**Important:** Replace `/absolute/path/to/mcp-router/src/mcp_server.py` with the actual absolute path to the file.

### 4. Alternative: Use Environment Variables

You can also set API keys in your shell environment or `.env` file:

```bash
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
```

Then reference the `.env` file in the MCP server script.

## Usage in Cursor

Once configured, Cursor will automatically:

1. **Analyze your queries** using the router
2. **Select the best model** based on:
   - Query complexity
   - Task type (reasoning, code, etc.)
   - Your preferences (cost, speed, quality)
3. **Route to the selected model** automatically

### Example Workflow

1. You type: "Explain how neural networks work"
   - Router analyzes → Reasoning task, high complexity
   - Router selects → Claude 3.5 Sonnet (best reasoning)
   - Cursor uses Claude 3.5 Sonnet for response

2. You type: "Write a Python function to sort a list"
   - Router analyzes → Code generation, low complexity
   - Router selects → GPT-4o-mini (fast, cost-effective)
   - Cursor uses GPT-4o-mini for response

3. You type: "Design a distributed system architecture"
   - Router analyzes → Complex reasoning task
   - Router selects → GPT-4o or Claude 3.5 Sonnet (high quality)
   - Cursor uses selected model for response

## Manual Model Selection

You can also manually request routing by using MCP tools in Cursor:

- `route_query`: Get routing recommendation without executing
- `execute_routed_query`: Route and execute query
- `list_models`: See all available models
- `get_routing_stats`: View routing statistics
- `analyze_query`: Analyze query characteristics

## Troubleshooting

### MCP Server Not Starting

1. Check Python path: Make sure `python3` is in your PATH
2. Check file path: Use absolute path to `mcp_server.py`
3. Check permissions: Ensure the script is executable

### Models Not Available

1. Check API keys: Ensure keys are set in environment or config
2. Check model registration: Verify models are registered in router
3. Check logs: Look for error messages in Cursor's MCP logs

### Routing Not Working

1. Verify MCP connection: Check Cursor's MCP status
2. Test router directly: Run `python main.py route "test query"`
3. Check configuration: Verify MCP server config is correct

## Advanced Configuration

### Custom Routing Strategies

You can customize routing by modifying the router configuration:

```python
# In mcp_server.py, customize the router
router = MCPRouter()

# Add custom models
router.register_model(ModelCapabilities(...))

# Modify routing logic in router.py
```

### Environment-Specific Models

Set different models for different environments:

```json
{
  "mcpServers": {
    "mcp-router": {
      "command": "python3",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "ROUTING_STRATEGY": "balanced"
      }
    }
  }
}
```

## Testing the Integration

1. **Test MCP Server Directly:**
   ```bash
   python3 src/mcp_server.py
   ```

2. **Test Router:**
   ```bash
   python3 main.py route "test query"
   ```

3. **Test in Cursor:**
   - Open Cursor
   - Check MCP status (should show "mcp-router" connected)
   - Try a query and observe model selection

## Benefits

- **Automatic Optimization**: Always uses the best model for each task
- **Cost Efficiency**: Routes to cheaper models when appropriate
- **Speed Optimization**: Uses faster models for simple tasks
- **Quality Focus**: Uses best models for complex reasoning
- **Transparency**: See which model was selected and why

## Next Steps

- Customize model configurations for your use case
- Add more models to the registry
- Implement custom routing strategies
- Monitor routing statistics to optimize performance

