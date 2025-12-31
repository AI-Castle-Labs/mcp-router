# Installation Guide for MCP Router

## Quick Install

### Option 1: Using pip with --user flag (Recommended)

```bash
cd mcp-router
pip3 install --user -r requirements.txt
pip3 install --user mcp
```

### Option 2: Using Virtual Environment (Best Practice)

```bash
cd mcp-router
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install mcp
```

### Option 3: Using pipx (macOS)

```bash
brew install pipx
pipx install mcp
```

## Verify Installation

```bash
python3 -c "from mcp.server import Server; print('MCP SDK installed')"
python3 -c "from src.router import MCPRouter; print('Router OK')"
```

## Setup Cursor Integration

After installing dependencies:

```bash
./scripts/setup_cursor.sh
```

Or manually configure Cursor's MCP settings.

## Troubleshooting

### "ModuleNotFoundError: No module named 'mcp'"

- Make sure you installed with `pip3 install --user mcp` or in a virtual environment
- Check Python path: `which python3`
- Try: `python3 -m pip install --user mcp`

### "Permission denied" errors

- Use `--user` flag: `pip3 install --user mcp`
- Or use virtual environment (Option 2 above)

### Cursor not connecting

- Check MCP server path is absolute
- Verify API keys are set
- Check Cursor logs for errors




