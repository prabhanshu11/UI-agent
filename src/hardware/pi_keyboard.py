"""Pi Zero 2W Bluetooth keyboard HTTP client.

Communicates with the Pi's keyboard API (Flask on port 8081) to send
keystrokes via Bluetooth HID to the target Windows machine.
"""

import requests
from typing import Optional


class PiKeyboardClient:
    """HTTP client for Pi Zero 2W keyboard API."""

    def __init__(self, host: str = '10.55.0.2', port: int = 8081, timeout: float = 2.0):
        self.base_url = f'http://{host}:{port}'
        self.timeout = timeout

    def health(self) -> dict:
        """Check keyboard API health.

        Returns:
            Health status dict, e.g. {"status": "healthy", "hid_ready": true, ...}
        """
        resp = requests.get(f'{self.base_url}/api/keyboard/health', timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def status(self) -> dict:
        """Check Bluetooth connection status.

        Returns:
            Status dict with connected state and device info.
        """
        resp = requests.get(f'{self.base_url}/api/keyboard/status', timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def key_press(self, key: str, modifiers: Optional[list[str]] = None) -> dict:
        """Press and release a key with optional modifiers.

        Args:
            key: Key name (e.g. 'a', 'enter', 'f1', 'space').
            modifiers: List of modifier names (e.g. ['ctrl'], ['ctrl', 'shift']).

        Returns:
            Response dict from API.
        """
        payload = {'key': key}
        if modifiers:
            payload['modifiers'] = modifiers
        resp = requests.post(
            f'{self.base_url}/api/keyboard/key',
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def type_text(self, text: str, delay_ms: int = 10) -> dict:
        """Type a string of text.

        Args:
            text: Text to type.
            delay_ms: Delay between keystrokes in milliseconds.

        Returns:
            Response dict from API.
        """
        resp = requests.post(
            f'{self.base_url}/api/keyboard/type',
            json={'text': text, 'delay_ms': delay_ms},
            timeout=self.timeout + len(text) * delay_ms / 1000,
        )
        resp.raise_for_status()
        return resp.json()

    def combo(self, keys: list[str], modifiers: Optional[list[str]] = None) -> dict:
        """Press a key combination (multiple keys + modifiers simultaneously).

        Args:
            keys: List of key names to press together.
            modifiers: List of modifier names.

        Returns:
            Response dict from API.
        """
        payload = {'keys': keys}
        if modifiers:
            payload['modifiers'] = modifiers
        resp = requests.post(
            f'{self.base_url}/api/keyboard/combo',
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def is_connected(self) -> bool:
        """Check if Pi keyboard is reachable and BT connected."""
        try:
            health = self.health()
            return health.get('status') in ('healthy', 'ok')
        except (requests.RequestException, KeyError):
            return False


if __name__ == '__main__':
    import sys

    client = PiKeyboardClient()

    if len(sys.argv) < 2:
        print("Usage: python pi_keyboard.py <command> [args...]")
        print("Commands:")
        print("  health         - Check API health")
        print("  status         - Check BT connection status")
        print("  key <key> [mod1 mod2...]  - Press key with modifiers")
        print("  type <text>    - Type text string")
        print("Examples:")
        print("  python pi_keyboard.py key enter")
        print("  python pi_keyboard.py key c ctrl")
        print("  python pi_keyboard.py type 'Hello World'")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'health':
        print(client.health())
    elif cmd == 'status':
        print(client.status())
    elif cmd == 'key':
        key = sys.argv[2]
        mods = sys.argv[3:] if len(sys.argv) > 3 else None
        print(client.key_press(key, mods))
    elif cmd == 'type':
        text = sys.argv[2]
        print(client.type_text(text))
    else:
        print(f"Unknown command: {cmd}")
