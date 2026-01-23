"""Screenshot Summarizer Agent - Describes screenshot content using vision LLMs.

Analyzes screenshots and generates structured summaries:
- What application/window is visible
- UI state and active elements
- What the user appears to be doing
- Text visible on screen (OCR-style)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml

from PIL import Image

from ..vision.llm_vision import LLMVision, VisionResult


# Supported formats for Claude/OpenAI vision
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB max
MAX_DIMENSION = 8192  # Max width/height


def validate_image_for_vision(image_path: Path) -> Tuple[bool, str, Optional[Path]]:
    """Validate and optionally fix image for vision model compatibility.

    Args:
        image_path: Path to image file

    Returns:
        (is_valid, error_or_warning, fixed_path) tuple
        If image was converted, fixed_path contains the new path
    """
    if not image_path.exists():
        return False, f"File not found: {image_path}", None

    # Check file size
    file_size = image_path.stat().st_size
    if file_size > MAX_IMAGE_SIZE:
        return False, f"File too large: {file_size / 1024 / 1024:.1f}MB (max {MAX_IMAGE_SIZE / 1024 / 1024}MB)", None

    if file_size == 0:
        return False, "Empty file", None

    # Check format
    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {suffix}. Supported: {SUPPORTED_FORMATS}", None

    # Validate actual image content
    try:
        with Image.open(image_path) as img:
            # Check dimensions
            if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
                return False, f"Image too large: {img.width}x{img.height} (max {MAX_DIMENSION})", None

            if img.width < 10 or img.height < 10:
                return False, f"Image too small: {img.width}x{img.height}", None

            # Check for corrupted images
            try:
                img.load()
            except Exception as e:
                return False, f"Corrupted image: {e}", None

            # Check color mode - convert if needed
            if img.mode not in ('RGB', 'RGBA', 'L'):
                # Need to convert
                fixed_path = image_path.with_suffix('.converted.png')
                if img.mode == 'P':
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
                img.save(fixed_path, 'PNG')
                return True, f"Converted from {img.mode} to RGB", fixed_path

    except Exception as e:
        return False, f"Cannot open image: {e}", None

    return True, "", None


def prepare_image_for_vision(image_path: Path) -> Tuple[Path, list[str]]:
    """Prepare image for vision model, with warnings.

    Args:
        image_path: Path to image

    Returns:
        (usable_path, warnings) tuple
    """
    warnings = []

    is_valid, message, fixed_path = validate_image_for_vision(image_path)

    if not is_valid:
        raise ValueError(f"Invalid image: {message}")

    if message:
        warnings.append(message)

    return fixed_path or image_path, warnings


class ScreenshotSummarizer:
    """Summarize screenshots using vision LLMs."""

    def __init__(
        self,
        agent_dir: Path,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        """Initialize summarizer.

        Args:
            agent_dir: Path to agent directory
            model: Vision model to use
            api_key: OpenRouter API key (or from env)
        """
        self.agent_dir = Path(agent_dir)
        self.task_file = self.agent_dir / "task.md"
        self.output_file = self.agent_dir / "output.md"

        # Initialize vision model (lazy - only when needed)
        self._model = model
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._vision: Optional[LLMVision] = None

    @property
    def vision(self) -> LLMVision:
        """Lazy-load vision model."""
        if self._vision is None:
            if not self._api_key:
                raise ValueError("OPENROUTER_API_KEY required for summarization")
            self._vision = LLMVision(api_key=self._api_key, model=self._model)
        return self._vision

    def summarize_screenshot(
        self,
        screenshot_path: Path,
        context: Optional[str] = None
    ) -> dict:
        """Generate summary for a single screenshot.

        Args:
            screenshot_path: Path to screenshot image
            context: Optional context about what task is being performed

        Returns:
            Summary dict with application, ui_state, visible_text, etc.
        """
        screenshot_path = Path(screenshot_path)

        # Validate image format before sending to vision model
        try:
            validated_path, warnings = prepare_image_for_vision(screenshot_path)
        except ValueError as e:
            return {
                "screenshot": str(screenshot_path),
                "timestamp": datetime.now().isoformat(),
                "error": f"Image validation failed: {str(e)}",
                "confidence": 0.0
            }

        task = f"Describe what is shown on this screen"
        if context:
            task = f"Describe what is shown on this screen. Context: {context}"

        try:
            result = self.vision.analyze_screen(validated_path, task)

            return {
                "screenshot": str(screenshot_path),
                "validated_path": str(validated_path) if validated_path != screenshot_path else None,
                "warnings": warnings if warnings else None,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "application": result.screen_description,
                    "elements": [
                        {
                            "name": e.name,
                            "type": e.element_type,
                            "position": {"x": e.x, "y": e.y},
                            "description": e.description
                        }
                        for e in result.elements[:10]  # Limit to 10 elements
                    ],
                    "element_count": len(result.elements)
                },
                "confidence": 0.85  # Default confidence
            }

        except Exception as e:
            return {
                "screenshot": str(screenshot_path),
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "confidence": 0.0
            }

    def quick_describe(self, screenshot_path: Path) -> str:
        """Get a quick one-line description of a screenshot.

        Args:
            screenshot_path: Path to screenshot

        Returns:
            Short description string
        """
        screenshot_path = Path(screenshot_path)

        # Validate image first
        try:
            validated_path, _ = prepare_image_for_vision(screenshot_path)
        except ValueError as e:
            return f"Invalid image: {str(e)}"

        try:
            result = self.vision.analyze_screen(
                validated_path,
                "Describe this screen in one sentence"
            )
            return result.screen_description or "Unknown screen"
        except Exception as e:
            return f"Error: {str(e)}"

    def detect_action(
        self,
        before_screenshot: Path,
        after_screenshot: Path
    ) -> dict:
        """Detect what action occurred between two screenshots.

        Args:
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action

        Returns:
            Dict describing the detected action
        """
        # For now, analyze after screenshot and infer action
        # TODO: Implement proper before/after comparison
        summary = self.summarize_screenshot(
            after_screenshot,
            context="Something changed from the previous state"
        )

        return {
            "before": str(before_screenshot),
            "after": str(after_screenshot),
            "action_detected": summary.get("summary", {}).get("application", "unknown"),
            "timestamp": datetime.now().isoformat()
        }

    def batch_summarize(
        self,
        screenshots: list[Path],
        output_dir: Optional[Path] = None,
        rate_limit_delay: float = 0.5
    ) -> list[dict]:
        """Summarize multiple screenshots.

        Args:
            screenshots: List of screenshot paths
            output_dir: Optional directory to save summaries
            rate_limit_delay: Seconds between API calls

        Returns:
            List of summary dicts
        """
        import time

        summaries = []
        for i, screenshot in enumerate(screenshots):
            print(f"Summarizing {i+1}/{len(screenshots)}: {screenshot.name}")

            summary = self.summarize_screenshot(screenshot)
            summaries.append(summary)

            # Rate limiting
            if i < len(screenshots) - 1:
                time.sleep(rate_limit_delay)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            summaries_file = output_dir / "summaries.json"
            with open(summaries_file, "w") as f:
                json.dump(summaries, f, indent=2)

        return summaries

    def check_for_task(self) -> Optional[dict]:
        """Check if there's a task to process."""
        if not self.task_file.exists():
            return None

        content = self.task_file.read_text()

        if "```yaml" in content:
            yaml_block = content.split("```yaml")[1].split("```")[0]
            try:
                task = yaml.safe_load(yaml_block)
                if task.get("status") == "pending":
                    return task
            except yaml.YAMLError:
                pass

        return None

    def write_output(self, results: dict):
        """Write results to output.md."""
        timestamp = datetime.now().isoformat()

        output = f"""# Summarization Results

**Generated**: {timestamp}
**Status**: completed

## Summary

- **Screenshots processed**: {results.get('count', 0)}
- **Success rate**: {results.get('success_rate', 0):.1%}

## Results

```yaml
summaries:
"""

        for summary in results.get('summaries', []):
            screenshot = summary.get('screenshot', 'unknown')
            desc = summary.get('summary', {}).get('application', 'unknown')
            output += f"""  - screenshot: "{screenshot}"
    description: "{desc}"
    element_count: {summary.get('summary', {}).get('element_count', 0)}
"""

        output += "```\n"

        self.output_file.write_text(output)

    def run_once(self) -> bool:
        """Check for task and process if available."""
        task = self.check_for_task()
        if not task:
            return False

        screenshots = [Path(p) for p in task.get('screenshots', [])]
        if not screenshots:
            # Check for directory
            screenshot_dir = task.get('screenshot_dir')
            if screenshot_dir:
                screenshots = list(Path(screenshot_dir).glob("*.png"))

        summaries = self.batch_summarize(screenshots)

        success_count = sum(1 for s in summaries if 'error' not in s)
        results = {
            'count': len(summaries),
            'success_rate': success_count / len(summaries) if summaries else 0,
            'summaries': summaries
        }

        self.write_output(results)

        # Mark task complete
        task_content = self.task_file.read_text()
        task_content = task_content.replace('status: pending', 'status: completed')
        self.task_file.write_text(task_content)

        return True


if __name__ == "__main__":
    # Test
    from pathlib import Path

    agent_dir = Path(__file__).parent.parent.parent / "agents" / "summarizer"
    summarizer = ScreenshotSummarizer(agent_dir)

    # Test on a captured screenshot
    test_image = Path("/tmp/capture_check.png")
    if test_image.exists():
        desc = summarizer.quick_describe(test_image)
        print(f"Quick description: {desc}")
    else:
        print("No test image found at /tmp/capture_check.png")
