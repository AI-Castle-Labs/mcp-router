# Quick Start: Cursor IDE Integration

This guide will help you integrate the MCP Router with Cursor IDE so it automatically selects the best model for each query.

## How It Works

```
Your Query → Router Analyzes → Best Model Selected → Cursor Uses That Model
```

## Step 1: Install Dependencies

```bash
cd mcp-router
pip install -r requirements.txt
```

## Step 2: No API Keys Needed!

**Important:** The router doesn't need API keys! It only recommends models to Cursor. Cursor uses its own configured API keys to make the actual API calls.

This means:
- ✅ No need to set API keys in the router
- ✅ Cursor already has your keys configured
- ✅ More secure - keys stay in Cursor

## Step 3: Run Setup Script

```bash
./scripts/setup_cursor.sh
```

This will:
- Find your Cursor config directory
- Configure the MCP server in `mcp.json`

## Step 4: Add to Agent Settings

After configuring the MCP server, you need to enable it in Cursor's agent settings:

### Option A: Via Settings UI
1. Open Cursor Settings (`Cmd+,` or `Ctrl+,`)
2. Go to `Features` → `Model Context Protocol`
3. Enable `mcp-router` server
4. Make sure "Use MCP Tools" is enabled in Agent settings

### Option B: Via Command Palette
1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Cursor: Open Agent Settings"
3. Enable MCP tools and select `mcp-router`

See `docs/AGENT_SETTINGS.md` for detailed instructions.

## Step 5: Verify Setup

This will:
- Find your Cursor config directory
- Create/update `mcp.json` with the router configuration
- Set up the MCP server path

## Step 4: Manual Configuration (Alternative)

If the script doesn't work, manually add to Cursor's MCP config:

**Location:**
- macOS: `~/Library/Application Support/Cursor/User/globalStorage/mcp.json`
- Linux: `~/.config/Cursor/User/globalStorage/mcp.json`
- Windows: `%APPDATA%\Cursor\User\globalStorage\mcp.json`

**Configuration:**
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

**Important:** Replace `/absolute/path/to/mcp-router/src/mcp_server.py` with the actual absolute path.

## Step 5: Restart Cursor

1. Close Cursor completely
2. Reopen Cursor
3. Check MCP status (should show "mcp-router" connected)

## Step 6: Test It!

1. Open Cursor chat
2. Type a query like: "Explain how neural networks work"
3. The router will automatically:
   - Analyze your query
   - Select the best model (e.g., Claude 3.5 Sonnet for reasoning)
   - Cursor will use that model

## Verification

To verify it's working:

1. **Check MCP Status:**
   - Open Cursor Settings → MCP
   - You should see "mcp-router" listed and connected

2. **Test Routing:**
   ```bash
   python3 main.py route "test query"
   ```

3. **Use MCP Tools in Cursor:**
   - In Cursor chat, you can use MCP tools:
     - `route_query` - Get routing recommendation
     - `execute_routed_query` - Route and execute
     - `list_models` - See available models

## Troubleshooting

### MCP Server Not Starting

1. **Check Python path:**
   ```bash
   which python3
   ```
   Make sure it's in your PATH

2. **Test MCP server directly:**
   ```bash
   python3 src/mcp_server.py
   ```
   Should start without errors

3. **Check file permissions:**
   ```bash
   chmod +x src/mcp_server.py
   ```

### Models Not Available

1. **Check router:**
   ```bash
   python3 main.py list
   ```
   Should show all registered models

2. **Note:** Router doesn't need API keys - it just recommends models. Cursor handles API calls.

### Cursor Not Connecting

1. **Check config file exists:**
   - Verify `mcp.json` is in the correct location
   - Check JSON syntax is valid

2. **Check Cursor logs:**
   - Look for MCP-related errors in Cursor's developer console

3. **Restart Cursor:**
   - Fully quit and restart Cursor

## Example Queries

Try these to see different models selected:

- **Reasoning:** "Explain quantum computing principles"
  → Should route to Claude 3.5 Sonnet or GPT-4o

- **Code Generation:** "Write a Python function to sort a list"
  → Should route to GPT-4o-mini or Claude 3 Haiku (fast, cost-effective)

- **Complex Architecture:** "Design a microservices architecture"
  → Should route to GPT-4o or Claude 3.5 Sonnet (high quality)

## Next Steps

- Customize routing strategies
- Add more models
- Monitor routing statistics
- Fine-tune model selection criteria

## Support

For issues or questions:
1. Check `docs/cursor_integration.md` for detailed docs
2. Test router directly: `python3 main.py route "your query"`
3. Check MCP server logs in Cursor

