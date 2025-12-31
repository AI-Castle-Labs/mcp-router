# Adding MCP Router to Cursor Agent Settings

This guide shows you how to add the MCP Router tool to Cursor's agent settings so it can intelligently route queries to the best model.

## Method 1: Via Cursor Settings UI (Recommended)

1. **Open Cursor Settings:**
   - Press `Cmd+,` (Mac) or `Ctrl+,` (Windows/Linux)
   - Or go to `Cursor` → `Settings` → `Features` → `Model Context Protocol`

2. **Add MCP Server:**
   - Click "Add MCP Server" or "Edit MCP Servers"
   - Server Name: `mcp-router`
   - Command: `python3`
   - Args: `/Users/ash/Desktop/RAGUIUC/internal-dev-tasks/mcp-router/src/mcp_server.py`
     (Replace with your actual path)
   - Environment Variables: Leave empty (no API keys needed)

3. **Enable Tools:**
   - In the same settings page, make sure MCP tools are enabled
   - The router tools should appear automatically:
     - `route_query` - Route queries to best model
     - `get_model_recommendation` - Get model recommendation
     - `list_models` - List available models
     - `analyze_query` - Analyze query characteristics
     - `get_routing_stats` - Get routing statistics

4. **Restart Cursor:**
   - Quit and reopen Cursor for changes to take effect

## Method 2: Via Configuration File

The MCP configuration is stored in:

**macOS:**
```
~/Library/Application Support/Cursor/User/globalStorage/mcp.json
```

**Windows:**
```
%APPDATA%\Cursor\User\globalStorage\mcp.json
```

**Linux:**
```
~/.config/Cursor/User/globalStorage/mcp.json
```

### Configuration Format

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

### Quick Setup Script

Run the setup script to automatically configure:

```bash
cd mcp-router
./scripts/setup_cursor.sh
```

## Method 3: Enable in Cursor Agent Settings

1. **Open Agent Settings:**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Cursor: Open Agent Settings"
   - Or go to `Cursor` → `Settings` → `Agent`

2. **Enable MCP Tools:**
   - Look for "Model Context Protocol" section
   - Enable "Use MCP Tools"
   - Select `mcp-router` from the list of available servers

3. **Configure Tool Usage:**
   - You can specify which tools the agent should use:
     - ✅ `route_query` - For automatic routing
     - ✅ `get_model_recommendation` - For model selection
     - ✅ `analyze_query` - For query analysis
   - Or leave all enabled for full functionality

## Verification

After setup, verify the MCP router is working:

1. **Check MCP Status:**
   - Open Cursor Settings
   - Go to `Features` → `Model Context Protocol`
   - You should see `mcp-router` listed with a green status indicator

2. **Test in Chat:**
   - Open Cursor chat (`Cmd+L` or `Ctrl+L`)
   - Ask: "What model would you use for code generation?"
   - The agent should use the `get_model_recommendation` tool

3. **Check Tool Availability:**
   - In chat, type `@mcp-router` to see available tools
   - Or check the MCP tools panel in settings

## How Cursor Agent Uses the Router

Once configured, Cursor's agent will:

1. **Automatically route queries** using `route_query` or `get_model_recommendation`
2. **Analyze query characteristics** using `analyze_query`
3. **Select optimal models** based on:
   - Task type (code, reasoning, writing, etc.)
   - Complexity (simple, moderate, complex)
   - Your preferences (cost, speed, quality)

## Troubleshooting

### MCP Server Not Showing Up

1. **Check Python Path:**
   ```bash
   which python3
   ```
   Make sure `python3` is in your PATH

2. **Check MCP SDK:**
   ```bash
   pip3 show mcp
   ```
   If not installed: `pip3 install --user mcp`

3. **Check Server Path:**
   Make sure the path in `mcp.json` is absolute and correct

4. **Check Logs:**
   - Open Cursor Settings → `Features` → `Model Context Protocol`
   - Click on `mcp-router` to see connection logs
   - Look for error messages

### Tools Not Available

1. **Restart Cursor** - Changes require a restart
2. **Check MCP Status** - Should show green/connected
3. **Verify Server Running** - Test manually:
   ```bash
   python3 /path/to/mcp-router/src/mcp_server.py
   ```

### Agent Not Using Router

1. **Enable MCP Tools** in Agent Settings
2. **Check Tool Permissions** - Make sure tools are enabled for the agent
3. **Try Explicit Mention** - Use `@mcp-router route_query` in chat

## Example Usage in Cursor Chat

Once configured, you can use it like this:

```
You: @mcp-router get_model_recommendation "Write a Python function to sort a list"

Agent: [Uses the tool and returns]
{
  "recommended_model": {
    "model_id": "gpt-4o-mini",
    "provider": "openai",
    "name": "GPT-4o-mini"
  },
  "confidence": 0.92,
  "reasoning": "Model is optimized for code_generation tasks..."
}
```

Or let the agent automatically use it:

```
You: What's the best model for debugging Python code?

Agent: [Automatically calls get_model_recommendation and responds]
Based on your query, I recommend GPT-4o-mini for debugging tasks...
```

## Next Steps

- ✅ MCP Router configured
- ✅ Tools available to Cursor agent
- ✅ Ready to use intelligent model routing!

The agent will now automatically select the best model for each query based on the router's recommendations.


