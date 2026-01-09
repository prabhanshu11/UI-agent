# UI-Agent

UI automation agent with browser control for shopping and web tasks.

## Features

- **PPE Control Loop**: Perceive-Plan-Execute cycle for UI automation
- **Browser Automation**: Headless Chromium via Playwright
- **API Server**: FastAPI server for remote browser control
- **Shopping Integration**: Amazon cart, search, and address verification

## Setup

```bash
# Install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium
```

## Usage

### API Server Mode (for Shopping Agent integration)

```bash
uv run python main.py api --port 8000
```

This starts a FastAPI server with endpoints for:
- `/health` - Health check
- `/browser/*` - Low-level browser control
- `/amazon/*` - Amazon-specific operations

### Controller Mode (PPE loop)

```bash
uv run python main.py controller --freq 1.0
```

## API Endpoints

### Browser Control
- `POST /browser/start` - Start headless browser
- `POST /browser/navigate` - Navigate to URL
- `POST /browser/click` - Click element
- `POST /browser/fill` - Fill form field
- `GET /browser/screenshot` - Take screenshot
- `GET /browser/content` - Get page content

### Amazon Shopping
- `GET /amazon/search?query=...` - Search products
- `POST /amazon/add-to-cart` - Add product to cart
- `GET /amazon/cart` - Get cart contents
- `POST /amazon/verify-address` - Verify delivery address

## Project Structure

```
UI-agent/
├── src/
│   ├── api/          # FastAPI server
│   ├── browser/      # Playwright browser controller
│   ├── core/         # PPE controller and types
│   ├── goals/        # Shopping goals
│   └── perception/   # Screen capture, input monitoring
├── tests/            # Unit tests
└── main.py           # Entry point
```

## Integration with Shopping Agent

The UI-Agent provides browser automation for the Shopping Agent project.
Run the API server on port 8000, and the Shopping Agent will connect to it
for Amazon cart operations.

## Commands

- `uv sync` - Install dependencies
- `uv run pytest` - Run tests
- `uv run python main.py api` - Start API server
- `uv run python main.py controller` - Start PPE loop
