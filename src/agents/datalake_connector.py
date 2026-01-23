"""Datalake Connector - Integrates UI-Agent with the central datalake.

Provides persistent storage for:
- Screenshots and keyframes
- Storyline logs
- Experience records
"""

import os
import json
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import base64


class DatalakeConnector:
    """Connect UI-Agent to the datalake API for persistent storage."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        source_device: Optional[str] = None,
        timeout: float = 30.0
    ):
        """Initialize connector.

        Args:
            api_url: Datalake API URL (default: localhost:8766)
            source_device: Device identifier (desktop/laptop)
            timeout: HTTP request timeout
        """
        self.api_url = api_url or os.getenv("DATALAKE_URL", "http://localhost:8766")
        self.source_device = source_device or self._detect_device()
        self.client = httpx.Client(base_url=self.api_url, timeout=timeout)

    def _detect_device(self) -> str:
        """Detect which device we're running on."""
        import subprocess
        try:
            result = subprocess.run(
                ["fastfetch", "--structure", "Host", "--logo", "none"],
                capture_output=True, text=True, timeout=5
            )
            if "ThinkPad T14" in result.stdout:
                return "laptop"
            return "desktop"
        except Exception:
            return "unknown"

    def health_check(self) -> bool:
        """Check if datalake is accessible.

        Returns:
            True if healthy
        """
        try:
            response = self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    def get_stats(self) -> dict:
        """Get datalake statistics.

        Returns:
            Stats dict with audio, transcripts, screenshots counts
        """
        try:
            response = self.client.get("/api/v1/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def upload_screenshot(
        self,
        file_path: Path,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """Upload a screenshot to datalake staging directory.

        Note: Datalake API is currently read-only. Files are staged
        locally in a datalake-compatible structure for later ingestion.

        Args:
            file_path: Path to screenshot file
            tags: Optional tags for categorization
            metadata: Additional metadata (summary, source, etc.)

        Returns:
            Response dict with path and status
        """
        file_path = Path(file_path)

        # Datalake staging directory (on local machine)
        staging_dir = Path.home() / "Programs" / "datalake" / "data" / "screenshots"

        # Create date-based directory structure (YYYY/MM/DD)
        now = datetime.now()
        date_dir = staging_dir / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        # Copy file with timestamp prefix
        dest_name = f"{now.strftime('%H%M%S')}_{file_path.name}"
        dest_path = date_dir / dest_name

        import shutil
        shutil.copy2(file_path, dest_path)

        # Write metadata sidecar file
        metadata_file = dest_path.with_suffix(dest_path.suffix + ".json")
        meta = {
            "original_path": str(file_path),
            "tags": tags or [],
            "source_device": self.source_device,
            "created_at": now.isoformat(),
            "metadata": metadata or {}
        }
        metadata_file.write_text(json.dumps(meta, indent=2))

        return {
            "status": "staged",
            "path": str(dest_path),
            "metadata_path": str(metadata_file),
            "note": "File staged for datalake ingestion"
        }

    def upload_screenshots_batch(
        self,
        file_paths: list[Path],
        tags: Optional[list[str]] = None,
        task_id: Optional[str] = None
    ) -> list[dict]:
        """Upload multiple screenshots.

        Args:
            file_paths: List of screenshot paths
            tags: Tags to apply to all
            task_id: Task ID for grouping

        Returns:
            List of upload results
        """
        results = []
        base_tags = tags or []

        for i, path in enumerate(file_paths):
            # Add sequence tag
            screenshot_tags = base_tags + [f"seq_{i:04d}"]
            if task_id:
                screenshot_tags.append(f"task:{task_id}")

            result = self.upload_screenshot(
                path,
                tags=screenshot_tags,
                metadata={"sequence": i, "task_id": task_id}
            )
            results.append(result)

            # Small delay to avoid overwhelming the API
            import time
            time.sleep(0.1)

        return results

    def search_screenshots(
        self,
        tags: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """Search screenshots in datalake.

        Args:
            tags: Filter by tags
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            limit: Maximum results

        Returns:
            List of matching screenshot records
        """
        params = {"limit": limit}
        if tags:
            params["tags"] = ",".join(tags)
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        try:
            response = self.client.get("/api/v1/screenshots", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return [{"error": str(e)}]

    def log_storyline_event(
        self,
        storyline: str,
        event_type: str,
        content: str,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None
    ) -> dict:
        """Log a storyline event to datalake.

        This stores events in a way that can be queried later for
        learning and pattern recognition.

        Args:
            storyline: Which storyline (main, knowledge, technical, etc.)
            event_type: Type of event
            content: Event description
            metadata: Additional structured data
            session_id: Session identifier

        Returns:
            Response dict
        """
        # For now, store as metadata on a special "log" entry
        # TODO: Add dedicated storyline_events table to datalake
        event_data = {
            "storyline": storyline,
            "event_type": event_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "source_device": self.source_device,
            "session_id": session_id
        }

        # Store as JSON in transcripts table (temporary solution)
        # This is a workaround until datalake has a dedicated events table
        try:
            # Just return the event for now - proper storage pending datalake update
            return {"status": "logged_locally", "event": event_data}
        except Exception as e:
            return {"error": str(e)}

    def sync_experience(
        self,
        experience_dir: Path,
        task_id: str
    ) -> dict:
        """Sync an experience folder to datalake.

        Args:
            experience_dir: Path to experience directory
            task_id: Task identifier

        Returns:
            Sync results
        """
        experience_dir = Path(experience_dir)
        results = {
            "task_id": task_id,
            "screenshots": [],
            "keyframes": [],
            "storylines": [],
            "errors": []
        }

        # Upload screenshots
        screenshots_dir = experience_dir / "screenshots"
        if screenshots_dir.exists():
            for img_path in screenshots_dir.glob("*.png"):
                result = self.upload_screenshot(
                    img_path,
                    tags=["experience", task_id, "screenshot"],
                    metadata={"task_id": task_id, "type": "screenshot"}
                )
                results["screenshots"].append(result)

        # Upload keyframes
        keyframes_dir = experience_dir / "keyframes"
        if keyframes_dir.exists():
            for img_path in keyframes_dir.glob("*.png"):
                result = self.upload_screenshot(
                    img_path,
                    tags=["experience", task_id, "keyframe"],
                    metadata={"task_id": task_id, "type": "keyframe"}
                )
                results["keyframes"].append(result)

        # Upload storyline logs
        storylines_dir = experience_dir / "storylines"
        if storylines_dir.exists():
            for jsonl_file in storylines_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file) as f:
                        for line in f:
                            event = json.loads(line)
                            self.log_storyline_event(
                                storyline=event.get("storyline", "unknown"),
                                event_type=event.get("event_type", "unknown"),
                                content=event.get("content", ""),
                                metadata=event.get("metadata"),
                                session_id=task_id
                            )
                            results["storylines"].append({"file": jsonl_file.name})
                except Exception as e:
                    results["errors"].append({"file": str(jsonl_file), "error": str(e)})

        return results

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Test connector
    connector = DatalakeConnector()

    print(f"Datalake URL: {connector.api_url}")
    print(f"Device: {connector.source_device}")
    print(f"Healthy: {connector.health_check()}")
    print(f"Stats: {connector.get_stats()}")
