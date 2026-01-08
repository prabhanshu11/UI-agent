import pytest
import shutil
from pathlib import Path
from core.log_manager import LogManager

@pytest.fixture
def temp_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    screenshot_dir = log_dir / "screenshots"
    log_dir.mkdir()
    screenshot_dir.mkdir()
    return log_dir

def test_prune_screenshots_count(temp_log_dir):
    manager = LogManager(log_dir=str(temp_log_dir), screenshot_dir=str(temp_log_dir / "screenshots"))
    
    # Create 10 dummy screenshots
    screenshot_dir = float_path = temp_log_dir / "screenshots"
    for i in range(10):
        (screenshot_dir / f"screen_{i}.png").touch()
        
    # Set limit to 5
    stats = manager.prune_logs(max_files=5)
    
    # Should have deleted 5
    assert stats['deleted_files'] == 5
    assert len(list(screenshot_dir.glob("*.png"))) == 5

def test_prune_size_limit(temp_log_dir):
    manager = LogManager(log_dir=str(temp_log_dir), screenshot_dir=str(temp_log_dir / "screenshots"))
    
    # Create a large file (1MB)
    large_file = temp_log_dir / "large.log"
    with open(large_file, "wb") as f:
        f.write(b"\0" * 1024 * 1024)
        
    # Check size (approx 1MB)
    assert manager._get_dir_size_mb(temp_log_dir) >= 1.0
    
    # Prune with limit 0.5MB
    stats = manager.prune_logs(max_size_mb=0.5)
    
    # Should delete the file
    assert stats['deleted_files'] > 0
    assert not large_file.exists()
