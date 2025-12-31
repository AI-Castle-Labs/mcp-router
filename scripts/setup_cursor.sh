#!/bin/bash
# Setup script for Cursor IDE integration

echo "🚀 Setting up MCP Router for Cursor IDE..."

# Get absolute path to mcp_server.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
MCP_SERVER_PATH="$PROJECT_ROOT/src/mcp_server.py"

echo "📍 MCP Server Path: $MCP_SERVER_PATH"

# Detect OS and set Cursor config path
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CURSOR_CONFIG_DIR="$HOME/Library/Application Support/Cursor/User/globalStorage"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    CURSOR_CONFIG_DIR="$HOME/.config/Cursor/User/globalStorage"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    CURSOR_CONFIG_DIR="$APPDATA/Cursor/User/globalStorage"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

echo "📁 Cursor Config Directory: $CURSOR_CONFIG_DIR"

# Create config directory if it doesn't exist
mkdir -p "$CURSOR_CONFIG_DIR"

# Check if mcp.json exists
MCP_CONFIG_FILE="$CURSOR_CONFIG_DIR/mcp.json"

if [ -f "$MCP_CONFIG_FILE" ]; then
    echo "✅ Found existing mcp.json"
    echo "📝 Please manually add the following to your mcp.json:"
else
    echo "📝 Creating new mcp.json..."
    cat > "$MCP_CONFIG_FILE" << EOF
{
  "version": "1.0",
  "mcpServers": {
    "mcp-router": {
      "command": "python3",
      "args": [
        "$MCP_SERVER_PATH"
      ],
      "env": {}
    }
  }
}
EOF
    echo "✅ Created mcp.json"
fi

echo ""
echo "📋 Configuration:"
echo "  Command: python3"
echo "  Args: $MCP_SERVER_PATH"
echo ""
echo "⚠️  Make sure to:"
echo "  1. Cursor already has API keys configured - no need to set them here"
echo "  2. Restart Cursor IDE"
echo "  3. Enable MCP Router in Agent Settings:"
echo "     - Open Cursor Settings (Cmd+,)"
echo "     - Go to Features → Model Context Protocol"
echo "     - Enable 'mcp-router' server"
echo "     - Enable 'Use MCP Tools' in Agent settings"
echo "  4. The router will recommend models, Cursor will use them with its own API keys"
echo ""
echo "📖 See docs/AGENT_SETTINGS.md for detailed instructions"
echo ""
echo "✅ Setup complete!"

