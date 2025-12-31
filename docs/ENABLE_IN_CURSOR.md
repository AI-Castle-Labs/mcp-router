# Quick Guide: Enable MCP Router in Cursor Agent Settings

## ✅ Step-by-Step Instructions

### 1. MCP Server is Already Configured ✅
The MCP server is already set up in your `mcp.json` file. You just need to enable it in Cursor's UI.

### 2. Enable in Cursor Settings

**Option A: Via Settings Menu (Easiest)**
1. Open Cursor
2. Press `Cmd+,` (Mac) or `Ctrl+,` (Windows/Linux) to open Settings
3. In the search bar, type: `MCP` or `Model Context Protocol`
4. Click on `Features` → `Model Context Protocol`
5. You should see `mcp-router` listed
6. Toggle it **ON** ✅
7. Make sure "Use MCP Tools" is enabled

**Option B: Via Command Palette**
1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type: `Cursor: Open Agent Settings`
3. Look for "Model Context Protocol" section
4. Enable `mcp-router`
5. Enable "Use MCP Tools"

### 3. Verify It's Working

1. **Check Status:**
   - In Settings → MCP, you should see `mcp-router` with a green indicator ✅
   - If it's red/yellow, check the logs for errors

2. **Test in Chat:**
   - Open Cursor chat (`Cmd+L` or `Ctrl+L`)
   - Type: `@mcp-router list_models`
   - You should see the available models listed

3. **Auto-Routing Test:**
   - Ask a question like: "What's the best model for code generation?"
   - The agent should automatically use the router tools

## 🎯 What Happens Next?

Once enabled, Cursor's agent will:
- ✅ Automatically analyze your queries
- ✅ Route to the best model based on query characteristics
- ✅ Use the recommended model with Cursor's own API keys

## 🔧 Troubleshooting

**MCP Router Not Showing:**
- Make sure you've restarted Cursor after running `scripts/setup_cursor.sh`
- Check that `mcp.json` exists in the correct location
- Verify Python path is correct

**Tools Not Available:**
- Restart Cursor completely
- Check MCP server status (should be green)
- Verify MCP SDK is installed: `pip3 show mcp`

**Agent Not Using Router:**
- Make sure "Use MCP Tools" is enabled in Agent settings
- Try explicitly mentioning: `@mcp-router get_model_recommendation "your query"`

## 📚 More Help

- See `docs/AGENT_SETTINGS.md` for detailed documentation
- See `docs/QUICKSTART_CURSOR.md` for full setup guide
- Check Cursor logs if something isn't working



