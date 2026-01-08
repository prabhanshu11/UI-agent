import os
import shutil
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class LogManager:
    """
    Manages application logs and screenshots, enforcing retention policies.
    """
    def __init__(self, log_dir: str = "logs", screenshot_dir: str = "logs/screenshots"):
        self.log_dir = Path(log_dir)
        self.screenshot_dir = Path(screenshot_dir)
        
        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def prune_logs(self, max_files: int = 50, max_size_mb: int = 500) -> Dict[str, int]:
        """
        Removes old files if limits are exceeded.
        :param max_files: Maximum number of screenshot files to keep.
        :param max_size_mb: Maximum total size of the logs directory in MB.
        :return: Summary of deleted files.
        """
        deleted_count = 0
        deleted_size_bytes = 0
        
        # 1. Prune Screenshots by Count (Keep newest)
        screenshots = sorted(self.screenshot_dir.glob("*.png"), key=os.path.getmtime)
        
        if len(screenshots) > max_files:
            to_delete = screenshots[:len(screenshots) - max_files]
            for f in to_delete:
                try:
                    size = f.stat().st_size
                    f.unlink()
                    deleted_count += 1
                    deleted_size_bytes += size
                    logger.debug(f"Deleted old screenshot: {f.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")

        # 2. Prune by Total Size (if still over limit)
        current_size_mb = self._get_dir_size_mb(self.log_dir)
        
        if current_size_mb > max_size_mb:
            logger.info(f"Log directory size ({current_size_mb:.2f} MB) exceeds limit ({max_size_mb} MB). Pruning aggressively.")
            # Simple strategy: Delete oldest files in the entire log tree until under limit
            all_files = sorted(self.log_dir.rglob("*"), key=lambda p: os.path.getmtime(p))
            
            for f in all_files:
                if not f.is_file(): continue
                
                if self._get_dir_size_mb(self.log_dir) <= max_size_mb:
                    break
                    
                try:
                    f.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")

        return {
            "deleted_files": deleted_count,
            "freed_bytes": deleted_size_bytes
        }

    def _get_dir_size_mb(self, path: Path) -> float:
        """Calculates total size of a directory in MB."""
        total = 0
        for p in path.rglob('*'):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024 * 1024)

    def get_log_history(self) -> List[Dict[str, Any]]:
        """Returns a list of available logs/screenshots."""
        history = []
        for f in self.screenshot_dir.glob("*.png"):
            history.append({
                "name": f.name,
                "path": str(f),
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                "size": f.stat().st_size
            })
        return sorted(history, key=lambda x: x['created'], reverse=True)
