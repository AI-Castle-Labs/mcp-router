# ✅ Setup Complete - Next Steps

## What Was Done

1. ✅ Created Cursor MCP configuration file at:
   `~/Library/Application Support/Cursor/User/globalStorage/mcp.json`

2. ✅ Configured MCP server path:
   `/Users/ash/Desktop/RAGUIUC/internal-dev-tasks/mcp-router/src/mcp_server.py`

## Next Steps

### 1. Install MCP SDK

You need to install the MCP Python SDK. Choose one method:

**Option A: Using --user flag (Recommended)**
```bash
cd /Users/ash/Desktop/RAGUIUC/internal-dev-tasks/mcp-router
pip3 install --user mcp
pip3 install --user -r requirements.txt
```

**Option B: Using Virtual Environment**
```bash
cd /Users/ash/Desktop/RAGUIUC/internal-dev-tasks/mcp-router
python3 -m venv venv
source venv/bin/activate
pip install mcp
pip install -r requirements.txt
```

**Option C: Using pipx**
```bash
brew install pipx
pipx install mcp
```

### 2. Set API Keys

Make sure your API keys are set as environment variables:

```bash
export OPENAI_API_KEY="your_openai_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"
```

Or add them to your `~/.zshrc` or `~/.bashrc`:
```bash
echo 'export OPENAI_API_KEY="your_key"' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY="your_key"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Test MCP Server

Test that the MCP server can start:

```bash
cd /Users/ash/Desktop/RAGUIUC/internal-dev-tasks/mcp-router
python3 src/mcp_server.py
```

It should start without errors (it will wait for stdio input, which is normal).

Press Ctrl+C to stop it.

### 4. Restart Cursor

1. **Quit Cursor completely** (Cmd+Q on macOS)
2. **Reopen Cursor**
3. **Check MCP Status:**
   - Open Cursor Settings
   - Go to MCP section
   - You should see "mcp-router" listed and connected

### 5. Verify Integration

In Cursor:
1. Open the chat/command palette
2. Try using MCP tools:
   - `route_query` - Get routing recommendation
   - `list_models` - See available models
   - `analyze_query` - Analyze query characteristics

## Troubleshooting

### MCP Server Won't Start

**Error: "ModuleNotFoundError: No module named 'mcp'"**
- Install MCP SDK: `pip3 install --user mcp`
- Or use virtual environment (see Option B above)

**Error: "Permission denied"**
- Use `--user` flag: `pip3 install --user mcp`
- Or check file permissions: `chmod +x src/mcp_server.py`

### Cursor Not Connecting

1. **Check config file exists:**
   ```bash
   cat ~/Library/Application\ Support/Cursor/User/globalStorage/mcp.json
   ```

2. **Verify Python path:**
   ```bash
   which python3
   ```
   Make sure it matches what's in mcp.json

3. **Check Cursor logs:**
   - Open Cursor Developer Tools
   - Look for MCP-related errors

4. **Verify API keys:**
   ```bash
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   ```

### Test Router Directly

Test the router without Cursor:

```bash
cd /Users/ash/Desktop/RAGUIUC/internal-dev-tasks/mcp-router
python3 main.py route "test query"
python3 main.py list
```

## Configuration File Location

Your Cursor MCP config is at:
```
~/Library/Application Support/Cursor/User/globalStorage/mcp.json
```

You can edit it manually if needed. The current configuration:

```json
{
  "version": "1.0",
  "mcpServers": {
    "mcp-router": {
      "command": "python3",
      "args": [
        "/Users/ash/Desktop/RAGUIUC/internal-dev-tasks/mcp-router/src/mcp_server.py"
      ],
      "env": {}
    }
  }
}
```

## Success Indicators

✅ MCP server starts without errors
✅ Cursor shows "mcp-router" in MCP settings
✅ You can use MCP tools in Cursor chat
✅ Router automatically selects models for queries

## Need Help?

- Check `docs/QUICKSTART_CURSOR.md` for quick reference
- See `docs/cursor_integration.md` for detailed docs
- Test router directly: `python3 main.py route "your query"`



