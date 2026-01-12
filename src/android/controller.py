"""Android device controller via ADB."""

import subprocess
import time
from pathlib import Path
from typing import Optional


class AndroidController:
    """Control Android device via ADB commands."""
    
    def __init__(
        self,
        device_serial: str | None = None,
        adb_path: str = "adb",
    ):
        """Initialize controller.
        
        Args:
            device_serial: Specific device serial (None for first device)
            adb_path: Path to adb executable
        """
        self.device_serial = device_serial
        self.adb_path = adb_path
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to device.
        
        Returns:
            True if connected successfully
        """
        try:
            result = self._run_adb(["devices"])
            lines = result.stdout.strip().split("\n")[1:]
            devices = [l.split()[0] for l in lines if l.strip() and "device" in l]
            
            if not devices:
                print("No Android devices connected")
                return False
            
            self.device_serial = self.device_serial or devices[0]
            self._connected = True
            print(f"Connected to: {self.device_serial}")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def _run_adb(self, args: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
        """Run ADB command."""
        cmd = [self.adb_path]
        if self.device_serial and "-s" not in args:
            cmd.extend(["-s", self.device_serial])
        cmd.extend(args)
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    
    def shell(self, command: str) -> str:
        """Run shell command on device.
        
        Args:
            command: Shell command
            
        Returns:
            Command output
        """
        result = self._run_adb(["shell", command])
        return result.stdout
    
    def tap(self, x: int, y: int) -> None:
        """Tap at coordinates."""
        self.shell(f"input tap {x} {y}")
    
    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 300,
    ) -> None:
        """Swipe gesture."""
        self.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")
    
    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        """Long press at coordinates."""
        self.shell(f"input swipe {x} {y} {x} {y} {duration_ms}")
    
    def key_event(self, keycode: int | str) -> None:
        """Send key event."""
        self.shell(f"input keyevent {keycode}")
    
    def text(self, text: str) -> None:
        """Type text."""
        escaped = text.replace(" ", "%s").replace("'", "\\'")
        self.shell(f"input text '{escaped}'")
    
    def back(self) -> None:
        """Press back button."""
        self.key_event(4)
    
    def home(self) -> None:
        """Press home button."""
        self.key_event(3)
    
    def enter(self) -> None:
        """Press enter."""
        self.key_event(66)
    
    def screenshot(self, local_path: str | Path) -> Path:
        """Take screenshot and pull to local.
        
        Args:
            local_path: Where to save screenshot
            
        Returns:
            Path to saved file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        remote = "/sdcard/screenshot_temp.png"
        self.shell(f"screencap -p {remote}")
        self._run_adb(["pull", remote, str(local_path)])
        self.shell(f"rm {remote}")
        
        return local_path
    
    def get_screen_size(self) -> tuple[int, int]:
        """Get screen resolution.
        
        Returns:
            (width, height)
        """
        output = self.shell("wm size")
        for line in output.split("\n"):
            if "size:" in line.lower():
                size_str = line.split(":")[-1].strip()
                w, h = size_str.split("x")
                return int(w), int(h)
        return 1080, 1920
    
    def is_screen_on(self) -> bool:
        """Check if screen is on."""
        output = self.shell("dumpsys power | grep 'Display Power'")
        return "ON" in output.upper()
    
    def wake_screen(self) -> None:
        """Wake up screen if off."""
        if not self.is_screen_on():
            self.key_event(26)  # POWER
            time.sleep(0.5)
    
    def launch_app(self, package: str, activity: str | None = None) -> None:
        """Launch an app.
        
        Args:
            package: Package name
            activity: Optional activity name
        """
        if activity:
            self.shell(f"am start -n {package}/{activity}")
        else:
            self.shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")
    
    def stop_app(self, package: str) -> None:
        """Force stop app."""
        self.shell(f"am force-stop {package}")
    
    def is_app_running(self, package: str) -> bool:
        """Check if app is running."""
        output = self.shell(f"pidof {package}")
        return bool(output.strip())
    
    def get_current_activity(self) -> str:
        """Get current foreground activity."""
        output = self.shell("dumpsys activity activities | grep mCurrentFocus")
        return output.strip()
    
    def pull_file(self, remote_path: str, local_path: str | Path) -> bool:
        """Pull file from device.
        
        Returns:
            True if successful
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._run_adb(["pull", remote_path, str(local_path)])
        return result.returncode == 0
    
    def push_file(self, local_path: str | Path, remote_path: str) -> bool:
        """Push file to device.
        
        Returns:
            True if successful
        """
        result = self._run_adb(["push", str(local_path), remote_path])
        return result.returncode == 0
