#!/usr/bin/env python3
"""Stdio MCP server exposing hardware control tools.

This server runs as a subprocess and communicates via JSON-RPC over stdin/stdout.
It provides tools for HDMI capture, ESP32 mouse control, Pi keyboard API,
and cursor detection — everything needed for autonomous Windows UI navigation.

Used by k380_agent.py via Claude Agent SDK's stdio MCP transport.
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from src.hardware.esp32_mouse import ESP32Mouse
from src.hardware.hdmi_sensor import HDMISensor
from src.hardware.cursor_locator import CursorLocator

# ── Hardware initialization ─────────────────────────────────────

mouse = ESP32Mouse("/dev/ttyUSB0", screen_width=1920, screen_height=1080)
mouse.start_heartbeat()
mouse.start_anti_sleep()

sensor = HDMISensor("/dev/video0")
locator = CursorLocator(mouse, sensor)

# Log to stderr (stdout is reserved for MCP protocol)
def log(msg):
    print(f"[HW-MCP] {msg}", file=sys.stderr, flush=True)

log("Hardware initialized: mouse, sensor, locator ready")

# ── Tool definitions ────────────────────────────────────────────

TOOLS = [
    {
        "name": "capture_screenshot",
        "description": "Capture the Windows screen via HDMI capture card. Returns the screenshot as an image. Use this to see what's on screen before and after every action. The image is 1280x720 but mouse coordinates are on a 1920x1080 grid, so multiply screenshot coords by 1.5 to get mouse coords.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mouse_move_to",
        "description": "Move mouse cursor to absolute screen position (x, y). Screen is 1920x1080. Use coordinates from screenshot multiplied by 1.5.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Target X position (0-1920)"},
                "y": {"type": "integer", "description": "Target Y position (0-1080)"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "mouse_move",
        "description": "Move mouse cursor by a relative amount. Positive dx=right, positive dy=down.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dx": {"type": "integer", "description": "Horizontal pixels to move"},
                "dy": {"type": "integer", "description": "Vertical pixels to move"},
            },
            "required": ["dx", "dy"],
        },
    },
    {
        "name": "mouse_click",
        "description": "Click a mouse button at the current cursor position.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
            },
        },
    },
    {
        "name": "get_cursor_position",
        "description": "Get the estimated cursor position. Approximate — use locate_cursor for precise detection.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "locate_cursor",
        "description": "Precisely detect cursor position using Lissajous curve sweep + video analysis. Slow (~5s) but accurate.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "type_text",
        "description": "Type text on Windows via the Pi Zero Bluetooth keyboard. Pi must be paired and connected.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "pi_keyboard_health",
        "description": "Check Pi Zero Bluetooth keyboard server health. Returns connection status, device class, profile info.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "scroll",
        "description": "Scroll the mouse wheel. Positive clicks = scroll down, negative = scroll up.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "clicks": {"type": "integer", "description": "Number of scroll clicks (positive=down, negative=up)"},
            },
            "required": ["clicks"],
        },
    },
    {
        "name": "wait",
        "description": "Wait for a specified number of seconds. Use after clicking to let the UI respond.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Seconds to wait (max 30)"},
            },
            "required": ["seconds"],
        },
    },
]

# ── Tool handlers ───────────────────────────────────────────────


def handle_capture_screenshot(args):
    frame = sensor.capture(settle_frames=3)
    img = Image.fromarray(frame)
    # Resize to 1280x720 to reduce token cost
    img = img.resize((1280, 720), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    x, y = mouse.get_position()
    log(f"Screenshot captured. Cursor at ({x}, {y})")
    return [
        {"type": "image", "data": b64, "mimeType": "image/jpeg"},
        {"type": "text", "text": f"Screenshot captured (1280x720). Cursor estimated at ({x}, {y}). Remember: multiply screenshot coords by 1.5 for mouse coords."},
    ]


def handle_mouse_move_to(args):
    x, y = args.get("x", 960), args.get("y", 540)
    actual_dx, actual_dy = mouse.move_to(x, y)
    new_x, new_y = mouse.get_position()
    log(f"move_to({x}, {y}) → now at ({new_x}, {new_y})")
    return [{"type": "text", "text": f"Moved to ({new_x}, {new_y}). Delta was ({actual_dx}, {actual_dy})"}]


def handle_mouse_move(args):
    dx, dy = args.get("dx", 0), args.get("dy", 0)
    actual_dx, actual_dy = mouse.move(dx, dy)
    x, y = mouse.get_position()
    log(f"move({dx}, {dy}) → now at ({x}, {y})")
    return [{"type": "text", "text": f"Moved ({actual_dx}, {actual_dy}). Cursor now at ({x}, {y})"}]


def handle_mouse_click(args):
    button = args.get("button", "left")
    mouse.click(button)
    x, y = mouse.get_position()
    log(f"click({button}) at ({x}, {y})")
    return [{"type": "text", "text": f"Clicked {button} at ({x}, {y})"}]


def handle_get_cursor_position(args):
    x, y = mouse.get_position()
    return [{"type": "text", "text": f"Estimated cursor position: ({x}, {y})"}]


def handle_locate_cursor(args):
    pos = locator.locate(verbose=False)
    if pos:
        mouse.set_position(pos.x, pos.y)
        log(f"Cursor detected at ({pos.x}, {pos.y}) corr={pos.correlation:.2f}")
        return [{"type": "text", "text": f"Cursor detected at ({pos.x}, {pos.y}) with correlation {pos.correlation:.2f}"}]
    else:
        log("Cursor not detected")
        return [{"type": "text", "text": "Cursor not detected. Try moving it to the center of the screen first."}]


def handle_type_text(args):
    import httpx
    text = args.get("text", "")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post("http://10.55.0.2:8081/api/keyboard/type", json={"text": text})
            log(f"type_text('{text}') → {resp.status_code}")
            return [{"type": "text", "text": f"Typed '{text}'. Response: {resp.status_code} {resp.text[:200]}"}]
    except Exception as e:
        return [{"type": "text", "text": f"Type failed: {e}. Is the Pi keyboard connected?"}]


def handle_pi_keyboard_health(args):
    import httpx
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get("http://10.55.0.2:8081/api/keyboard/health")
            log(f"pi_health → {resp.status_code}")
            return [{"type": "text", "text": f"Pi keyboard health: {resp.text}"}]
    except Exception as e:
        return [{"type": "text", "text": f"Pi health check failed: {e}"}]


def handle_scroll(args):
    clicks = args.get("clicks", 0)
    mouse.ser.write(f"SCROLL:{clicks}\n".encode())
    mouse.ser.flush()
    time.sleep(0.05)
    log(f"scroll({clicks})")
    return [{"type": "text", "text": f"Scrolled {clicks} clicks"}]


def handle_wait(args):
    seconds = min(args.get("seconds", 1.0), 30.0)
    time.sleep(seconds)
    return [{"type": "text", "text": f"Waited {seconds}s"}]


TOOL_HANDLERS = {
    "capture_screenshot": handle_capture_screenshot,
    "mouse_move_to": handle_mouse_move_to,
    "mouse_move": handle_mouse_move,
    "mouse_click": handle_mouse_click,
    "get_cursor_position": handle_get_cursor_position,
    "locate_cursor": handle_locate_cursor,
    "type_text": handle_type_text,
    "pi_keyboard_health": handle_pi_keyboard_health,
    "scroll": handle_scroll,
    "wait": handle_wait,
}

# ── MCP Protocol Loop ──────────────────────────────────────────


def respond(req_id, result):
    """Send a JSON-RPC success response."""
    msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def respond_error(req_id, code, message):
    """Send a JSON-RPC error response."""
    msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def main():
    log("MCP server starting, waiting for requests on stdin...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log(f"Invalid JSON: {line[:100]}")
            continue

        method = req.get("method", "")
        req_id = req.get("id")

        if method == "initialize":
            respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hardware", "version": "1.0.0"},
            })
            log("Initialized")

        elif method == "notifications/initialized":
            pass  # No response needed for notifications

        elif method == "tools/list":
            respond(req_id, {"tools": TOOLS})
            log(f"Listed {len(TOOLS)} tools")

        elif method == "tools/call":
            tool_name = req.get("params", {}).get("name", "")
            tool_args = req.get("params", {}).get("arguments", {})
            log(f"Calling tool: {tool_name}({tool_args})")

            handler = TOOL_HANDLERS.get(tool_name)
            if handler:
                try:
                    content = handler(tool_args)
                    respond(req_id, {"content": content})
                except Exception as e:
                    log(f"Tool error: {e}")
                    respond(req_id, {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True})
            else:
                respond_error(req_id, -32601, f"Unknown tool: {tool_name}")

        elif req_id is not None:
            respond_error(req_id, -32601, f"Unknown method: {method}")

    log("stdin closed, shutting down")
    mouse.close()


if __name__ == "__main__":
    main()
