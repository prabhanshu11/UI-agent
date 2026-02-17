"""WindowsNavigator — perceive → find → act → verify loop with recovery.

Executes declarative NavigationStep sequences on a Windows machine controlled
via ESP32 BLE HID mouse + HDMI capture card. Each step is logged to the
StorylineLogger on appropriate tracks and wrapped in OTel spans.

Recovery system:
    When a step fails verification, the navigator orients (fresh screenshot),
    generates hypotheses about what went wrong, dispatches parallel Claude
    Agent SDK instances to evaluate each hypothesis, and acts on the best one.
    This is logged as a "yellow alert" — the experience continues.
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image

from src.agents.storyline_logger import StorylineLogger, StorylineType
from src.hardware.cursor_locator import CursorLocator
from src.hardware.esp32_mouse import ESP32Mouse
from src.hardware.hdmi_sensor import HDMISensor
from src.vision.llm_vision import LLMVision, UIElement

from .types import (
    Hypothesis,
    NavigationStep,
    StepAction,
    StepOutcome,
    StepResult,
)

# Where hypothesis prompt templates live
_HYPOTHESES_DIR = Path(__file__).parent / "hypotheses"

# Max parallel Claude CLI subprocesses for hypothesis evaluation
_MAX_HYPOTHESIS_WORKERS = 3

# Minimum confidence to act on a hypothesis automatically
_AUTO_ACT_CONFIDENCE = 0.7


class WindowsNavigator:
    """Navigates Windows UI via perceive→act→verify loop with intelligent recovery.

    Args:
        mouse: ESP32Mouse for HID control
        sensor: HDMISensor for HDMI capture
        locator: CursorLocator for Lissajous-based cursor detection
        vision: LLMVision for UI element finding
        logger: StorylineLogger for multi-track narrative logging
        screenshot_dir: Where to save step screenshots
    """

    def __init__(
        self,
        mouse: ESP32Mouse,
        sensor: HDMISensor,
        locator: CursorLocator,
        vision: LLMVision,
        logger: StorylineLogger,
        screenshot_dir: Optional[Path] = None,
    ):
        self.mouse = mouse
        self.sensor = sensor
        self.locator = locator
        self.vision = vision
        self.logger = logger
        self.screenshot_dir = screenshot_dir or Path(
            logger.experience_dir / "screenshots"
        )
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._screenshot_counter = 0

    def run_experience(
        self,
        name: str,
        steps: list[NavigationStep],
    ) -> list[StepResult]:
        """Execute a full navigation experience.

        Phase 1: Establish control (reconnect, heartbeat, Lissajous locate)
        Phase 2: Navigate through each step
        Phase 3: Report results and save narrative

        Args:
            name: Experience name (e.g., "pair_k380")
            steps: Ordered list of navigation steps

        Returns:
            List of StepResults, one per step
        """
        experience_span = self.logger.start_experience(name, {
            "experience.steps": len(steps),
            "experience.type": "navigation",
        })

        results: list[StepResult] = []

        try:
            # Phase 1: Establish control
            self._establish_control()

            # Phase 2: Navigate
            for i, step in enumerate(steps):
                self.logger.log_main(
                    "step_start",
                    f"Step {i + 1}/{len(steps)}: {step.name}",
                    agent_name="navigator",
                )

                result = self.execute_step(step)
                results.append(result)

                if result.outcome == StepOutcome.SUCCESS:
                    self.logger.log_main(
                        "step_complete",
                        f"Step {i + 1} succeeded: {step.name}",
                        agent_name="navigator",
                    )
                elif result.outcome == StepOutcome.YELLOW_ALERT:
                    self.logger.log_main(
                        "yellow_alert",
                        f"Step {i + 1} recovered via hypothesis: {step.name}",
                        agent_name="navigator",
                    )
                else:
                    if step.optional:
                        self.logger.log_main(
                            "step_skipped",
                            f"Optional step {i + 1} failed, continuing: {step.name}",
                            agent_name="navigator",
                        )
                    else:
                        self.logger.log_main(
                            "step_failed",
                            f"Step {i + 1} failed ({result.outcome.value}): {step.name}",
                            metadata={"error": result.error},
                            agent_name="navigator",
                        )
                        break  # Non-optional failure — abort sequence

            # Phase 3: Report
            self.logger.log_main("experience_complete", f"Experience '{name}' finished", metadata={
                "total_steps": len(steps),
                "completed": sum(1 for r in results if r.outcome in (StepOutcome.SUCCESS, StepOutcome.YELLOW_ALERT)),
                "failed": sum(1 for r in results if r.outcome not in (StepOutcome.SUCCESS, StepOutcome.YELLOW_ALERT)),
            }, agent_name="navigator")

        finally:
            self.logger.end_step(experience_span, status=(
                "OK" if all(
                    r.outcome in (StepOutcome.SUCCESS, StepOutcome.YELLOW_ALERT)
                    for r in results
                ) else "ERROR"
            ))
            self.logger.save_narrative()

        return results

    def execute_step(self, step: NavigationStep) -> StepResult:
        """Execute a single navigation step: perceive → find → act → verify.

        Args:
            step: The navigation step to execute

        Returns:
            StepResult with outcome and metadata
        """
        step_span = self.logger.start_step(step.name, attributes={
            "step.target": step.target,
            "step.action": step.action.value,
            "step.optional": step.optional,
        })

        start_time = time.monotonic()
        element_found: UIElement | None = None
        position_clicked: tuple[int, int] | None = None
        screenshot_before: str | None = None
        screenshot_after: str | None = None
        attempts = 0
        outcome = StepOutcome.TIMEOUT
        error: str | None = None
        hypotheses: list[dict] | None = None

        try:
            # Pre-delay
            if step.pre_delay > 0:
                time.sleep(step.pre_delay)

            # PERCEIVE: capture current screen state
            frame_before = self.sensor.capture()
            screenshot_before = self._save_screenshot(frame_before, f"{step.name}_before")
            self.logger.log_perception(
                "screenshot",
                f"Captured screen before: {step.name}",
                screenshot_path=screenshot_before,
                agent_name="navigator",
            )

            if step.action == StepAction.WAIT:
                # WAIT action: just wait for timeout and verify
                time.sleep(step.timeout)
                if step.verify:
                    frame_after = self.sensor.capture()
                    screenshot_after = self._save_screenshot(frame_after, f"{step.name}_after")
                    if self._verify_screen(frame_after, step.verify):
                        outcome = StepOutcome.SUCCESS
                    else:
                        outcome = StepOutcome.UNEXPECTED_SCREEN
                        error = f"Verification failed after wait: {step.verify}"
                else:
                    outcome = StepOutcome.SUCCESS
            else:
                # FIND: locate target element via LLM vision
                for attempt in range(step.retries):
                    attempts = attempt + 1
                    self.logger.log_perception(
                        "element_search",
                        f"Looking for '{step.target}' (attempt {attempts}/{step.retries})",
                        agent_name="navigator",
                    )

                    element_found = self.vision.find_element(
                        Image.fromarray(frame_before), step.target
                    )

                    if element_found is not None:
                        self.logger.log_perception(
                            "element_found",
                            f"Found '{step.target}' at ({element_found.center_x}, {element_found.center_y}), confidence={element_found.confidence:.2f}",
                            agent_name="navigator",
                        )
                        break

                    # Re-capture for next attempt (screen may have changed)
                    time.sleep(1.0)
                    frame_before = self.sensor.capture()

                if element_found is None:
                    outcome = StepOutcome.ELEMENT_NOT_FOUND
                    error = f"Could not find '{step.target}' after {attempts} attempts"
                else:
                    # ACT: perform the action at the element's location
                    target_x, target_y = element_found.center_x, element_found.center_y
                    position_clicked = (target_x, target_y)

                    self.logger.log_technical(
                        "click",
                        f"Clicking '{step.target}' at ({target_x}, {target_y})",
                        agent_name="navigator",
                    )

                    click_success = self._perform_action(
                        step.action, target_x, target_y
                    )

                    if not click_success:
                        outcome = StepOutcome.CLICK_NO_EFFECT
                        error = f"Click verification failed at ({target_x}, {target_y})"
                    else:
                        # Post-delay
                        if step.post_delay > 0:
                            time.sleep(step.post_delay)

                        # VERIFY: check screen matches expected state
                        if step.verify:
                            frame_after = self.sensor.capture()
                            screenshot_after = self._save_screenshot(
                                frame_after, f"{step.name}_after"
                            )
                            self.logger.log_perception(
                                "screenshot",
                                f"Captured screen after: {step.name}",
                                screenshot_path=screenshot_after,
                                agent_name="navigator",
                            )

                            if self._verify_screen(frame_after, step.verify):
                                outcome = StepOutcome.SUCCESS
                            else:
                                outcome = StepOutcome.UNEXPECTED_SCREEN
                                error = f"Screen doesn't match: {step.verify}"
                        else:
                            outcome = StepOutcome.SUCCESS

            # If verification failed, attempt recovery
            if outcome in (StepOutcome.UNEXPECTED_SCREEN, StepOutcome.CLICK_NO_EFFECT):
                recovery_result = self.attempt_recovery(
                    step, screenshot_after or screenshot_before, position_clicked
                )
                if recovery_result is not None:
                    hypotheses = recovery_result["hypotheses"]
                    if recovery_result["recovered"]:
                        outcome = StepOutcome.YELLOW_ALERT
                        # Re-capture after recovery
                        frame_final = self.sensor.capture()
                        screenshot_after = self._save_screenshot(
                            frame_final, f"{step.name}_recovered"
                        )

        except Exception as e:
            outcome = StepOutcome.TIMEOUT
            error = str(e)
            self.logger.log_technical(
                "error",
                f"Exception during step '{step.name}': {e}",
                error=str(e),
                agent_name="navigator",
            )

        elapsed = time.monotonic() - start_time

        self.logger.end_step(step_span, status=(
            "OK" if outcome in (StepOutcome.SUCCESS, StepOutcome.YELLOW_ALERT) else "ERROR"
        ), attributes={
            "step.outcome": outcome.value,
            "step.attempts": attempts,
            "step.elapsed_s": round(elapsed, 2),
        })

        return StepResult(
            step=step,
            outcome=outcome,
            element_found=element_found,
            position_clicked=position_clicked,
            screenshot_before=screenshot_before,
            screenshot_after=screenshot_after,
            attempts=attempts,
            elapsed_s=elapsed,
            hypotheses=hypotheses,
            error=error,
        )

    def attempt_recovery(
        self,
        step: NavigationStep,
        screenshot_path: str | None,
        position_clicked: tuple[int, int] | None,
    ) -> dict | None:
        """Orient → hypothesize → dispatch → report recovery cycle.

        Args:
            step: The step that failed
            screenshot_path: Path to the failure screenshot
            position_clicked: Where we tried to click (if applicable)

        Returns:
            Dict with "recovered" bool and "hypotheses" list, or None if
            no screenshot available.
        """
        if screenshot_path is None:
            return None

        recovery_span = self.logger.start_step(
            f"recovery:{step.name}",
            attributes={"recovery.for_step": step.name},
        )

        try:
            # 1. ORIENT: fresh screenshot + full screen analysis
            frame = self.sensor.capture()
            orient_screenshot = self._save_screenshot(frame, f"{step.name}_orient")
            self.logger.log_perception(
                "screenshot",
                f"Recovery orientation capture",
                screenshot_path=orient_screenshot,
                agent_name="navigator",
            )

            screen_analysis = self.vision.analyze_screen(
                Image.fromarray(frame),
                task=f"Describe the current screen state. We expected to see: {step.verify}"
            )
            self.logger.log_perception(
                "screen_analysis",
                f"Screen shows: {screen_analysis.screen_description}",
                agent_name="navigator",
            )
            self.logger.log_knowledge(
                f"Expected '{step.verify}' but see '{screen_analysis.screen_description}'",
                applies_to=["navigation", step.name],
                confidence=0.6,
                agent_name="navigator",
            )

            # 2. HYPOTHESIZE: select relevant hypothesis prompts
            hypotheses = self._select_hypotheses(
                step, orient_screenshot, position_clicked
            )

            # 3. DISPATCH: parallel Agent SDK calls
            for h in hypotheses:
                self.logger.log_agent(
                    f"hypothesis:{h.name}",
                    "dispatched",
                    f"Dispatched hypothesis: {h.name}",
                    task=f"Evaluate {h.name} hypothesis",
                    agent_name="navigator",
                )

            evaluated = self._dispatch_hypotheses(hypotheses)

            # 4. COLLECT: gather results
            hypothesis_results = []
            best_hypothesis: Hypothesis | None = None
            best_confidence = 0.0

            for h in evaluated:
                self.logger.log_agent(
                    f"hypothesis:{h.name}",
                    "completed",
                    f"Hypothesis '{h.name}': confidence={h.confidence:.2f}",
                    result=h.result,
                    agent_name="navigator",
                )
                hypothesis_results.append({
                    "name": h.name,
                    "confidence": h.confidence,
                    "result": h.result,
                    "recommended_action": h.recommended_action,
                })
                if h.confidence > best_confidence:
                    best_confidence = h.confidence
                    best_hypothesis = h

            # 5. YELLOW ALERT: log the recovery attempt
            recovered = False
            if best_hypothesis and best_confidence >= _AUTO_ACT_CONFIDENCE:
                self.logger.log_main(
                    "yellow_alert",
                    f"Recovery: acting on '{best_hypothesis.name}' "
                    f"(confidence={best_confidence:.2f}): "
                    f"{best_hypothesis.recommended_action}",
                    metadata={"hypotheses": hypothesis_results},
                    agent_name="navigator",
                )
                self.logger.log_knowledge(
                    f"Recovery insight from '{best_hypothesis.name}': {best_hypothesis.result}",
                    applies_to=["recovery", step.name],
                    confidence=best_confidence,
                    agent_name="navigator",
                )
                # 6. ACT on best hypothesis (if we can)
                recovered = self._act_on_hypothesis(best_hypothesis, step)
            else:
                self.logger.log_main(
                    "yellow_alert",
                    f"Recovery: no high-confidence hypothesis "
                    f"(best={best_confidence:.2f}). Escalating to user.",
                    metadata={"hypotheses": hypothesis_results},
                    agent_name="navigator",
                )
                self.logger.log_user(
                    "escalation",
                    f"Step '{step.name}' failed and recovery is uncertain. "
                    f"Best hypothesis: {best_hypothesis.name if best_hypothesis else 'none'} "
                    f"({best_confidence:.2f}). Please check the screen.",
                    agent_name="navigator",
                )

            return {"recovered": recovered, "hypotheses": hypothesis_results}

        finally:
            self.logger.end_step(recovery_span, status="OK" if recovered else "ERROR")

    # ── Internal methods ──────────────────────────────────────────

    def _establish_control(self):
        """Phase 1: Ensure we have mouse control and know cursor position."""
        control_span = self.logger.start_step("establish_control", attributes={
            "phase": "control",
        })

        try:
            # Reconnect BLE if needed
            self.logger.log_technical(
                "ble_check",
                "Checking ESP32 mouse connection",
                agent_name="navigator",
            )
            status = self.mouse.status()
            if status != "CONNECTED":
                self.logger.log_technical(
                    "ble_reconnect",
                    f"Mouse status: {status}, attempting reconnect",
                    agent_name="navigator",
                )
                self.mouse.reconnect()

            # Start heartbeat
            self.mouse.start_heartbeat()
            self.logger.log_technical(
                "heartbeat_start",
                "BLE heartbeat started",
                agent_name="navigator",
            )

            # Locate cursor via Lissajous
            self.logger.log_perception(
                "lissajous_start",
                "Starting Lissajous cursor detection",
                agent_name="navigator",
            )
            pos = self.locator.locate(verbose=True)
            if pos:
                self.logger.log_perception(
                    "cursor_detected",
                    f"Cursor at ({pos.x}, {pos.y}), correlation={pos.correlation:.2f}",
                    agent_name="navigator",
                )
                self.logger.end_step(control_span, status="OK", attributes={
                    "cursor.x": pos.x,
                    "cursor.y": pos.y,
                    "cursor.correlation": pos.correlation,
                })
            else:
                self.logger.log_perception(
                    "cursor_not_found",
                    "Lissajous sweep failed to find cursor",
                    agent_name="navigator",
                )
                self.logger.end_step(control_span, status="ERROR")
                raise RuntimeError("Could not locate cursor on HDMI screen")

        except Exception:
            self.logger.end_step(control_span, status="ERROR")
            raise

    def _perform_action(
        self, action: StepAction, x: int, y: int
    ) -> bool:
        """Perform click/action at coordinates using CursorLocator.verified_click.

        Returns True if the action was performed (verification passed).
        """
        if action == StepAction.CLICK:
            return self.locator.verified_click(x, y, button="left")
        elif action == StepAction.RIGHT_CLICK:
            return self.locator.verified_click(x, y, button="right")
        elif action == StepAction.DOUBLE_CLICK:
            # Double-click: two quick clicks at the same spot
            success = self.locator.verified_click(x, y, button="left", verify=False)
            if success:
                time.sleep(0.05)
                self.mouse.click("left")
            return success
        elif action == StepAction.WAIT:
            return True  # No action needed
        return False

    def _verify_screen(self, frame, expected: str) -> bool:
        """Ask LLM vision if the screen matches the expected state.

        Returns True if the LLM confirms the expected state is visible.
        """
        result = self.vision.analyze_screen(
            Image.fromarray(frame),
            task=f"Does this screen show: {expected}? Answer with the first word being YES or NO, then explain briefly."
        )
        # Check if the LLM response starts with YES
        description = result.screen_description.strip().upper()
        raw = result.raw_response
        try:
            content = raw["choices"][0]["message"]["content"].strip().upper()
            return content.startswith("YES")
        except (KeyError, IndexError):
            return "YES" in description[:20]

    def _save_screenshot(self, frame, label: str) -> str:
        """Save a screenshot frame to disk and return the path."""
        self._screenshot_counter += 1
        filename = f"{self._screenshot_counter:03d}_{label}.png"
        path = self.screenshot_dir / filename
        Image.fromarray(frame).save(str(path))
        return str(path)

    def _select_hypotheses(
        self,
        step: NavigationStep,
        screenshot: str,
        position_clicked: tuple[int, int] | None,
    ) -> list[Hypothesis]:
        """Select relevant hypothesis prompts based on failure context."""
        hypotheses = []

        # Always check for unexpected dialog
        hypotheses.append(Hypothesis(
            name="dialog_appeared",
            prompt_file=str(_HYPOTHESES_DIR / "dialog_appeared.md"),
            screenshot=screenshot,
        ))

        # If we clicked somewhere, check if the click missed
        if position_clicked is not None:
            hypotheses.append(Hypothesis(
                name="click_missed",
                prompt_file=str(_HYPOTHESES_DIR / "click_missed.md"),
                screenshot=screenshot,
            ))

            # Check for cursor drift
            hypotheses.append(Hypothesis(
                name="cursor_drifted",
                prompt_file=str(_HYPOTHESES_DIR / "cursor_drifted.md"),
                screenshot=screenshot,
            ))

        # Always check for wrong screen
        hypotheses.append(Hypothesis(
            name="wrong_screen",
            prompt_file=str(_HYPOTHESES_DIR / "wrong_screen.md"),
            screenshot=screenshot,
        ))

        # Limit to 3 for parallel dispatch efficiency
        return hypotheses[:3]

    def _dispatch_hypotheses(
        self, hypotheses: list[Hypothesis]
    ) -> list[Hypothesis]:
        """Dispatch hypotheses to parallel Claude CLI subprocesses.

        Uses the same pattern as l3_analyzer.py — subprocess calls to
        `claude -p` with each hypothesis prompt + screenshot path.
        """
        def evaluate_one(hypothesis: Hypothesis) -> Hypothesis:
            try:
                # Read the prompt template
                template = Path(hypothesis.prompt_file).read_text()

                # Build the full prompt with screenshot
                prompt = f"""{template}

Read the image file at {hypothesis.screenshot} and apply the analysis described above."""

                result = subprocess.run(
                    ["claude", "-p", prompt],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode == 0 and result.stdout.strip():
                    response = result.stdout.strip()
                    hypothesis.result = response

                    # Try to parse JSON from response
                    try:
                        # Handle markdown code blocks
                        text = response
                        if "```" in text:
                            text = text.split("```")[1]
                            if text.startswith("json"):
                                text = text[4:]
                        data = json.loads(text.strip())
                        hypothesis.confidence = data.get("confidence", 0.0)
                        hypothesis.recommended_action = data.get(
                            "recommended_action", None
                        )
                    except (json.JSONDecodeError, IndexError):
                        hypothesis.confidence = 0.3  # Low confidence if unparseable

            except subprocess.TimeoutExpired:
                hypothesis.result = "Timeout"
                hypothesis.confidence = 0.0
            except Exception as e:
                hypothesis.result = f"Error: {e}"
                hypothesis.confidence = 0.0

            return hypothesis

        with ThreadPoolExecutor(max_workers=_MAX_HYPOTHESIS_WORKERS) as executor:
            futures = {
                executor.submit(evaluate_one, h): h for h in hypotheses
            }
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _act_on_hypothesis(
        self, hypothesis: Hypothesis, step: NavigationStep
    ) -> bool:
        """Attempt to act on a recovery hypothesis.

        Returns True if the action was taken and seems to have worked.
        """
        action = hypothesis.recommended_action
        if action is None:
            return False

        self.logger.log_technical(
            "recovery_action",
            f"Acting on hypothesis '{hypothesis.name}': {action}",
            agent_name="navigator",
        )

        # Parse known action types
        if action.startswith("click_button:"):
            button_label = action.split(":", 1)[1]
            # Find and click the button
            frame = self.sensor.capture()
            element = self.vision.find_element(
                Image.fromarray(frame), button_label
            )
            if element:
                return self.locator.verified_click(
                    element.center_x, element.center_y
                )
            return False

        elif action == "retry_click_with_offset":
            # Try clicking the original target again with fresh detection
            frame = self.sensor.capture()
            element = self.vision.find_element(
                Image.fromarray(frame), step.target
            )
            if element:
                return self.locator.verified_click(
                    element.center_x, element.center_y
                )
            return False

        elif action == "recalibrate_position":
            # Re-run Lissajous to fix cursor drift
            pos = self.locator.locate(verbose=False)
            if pos:
                self.logger.log_perception(
                    "cursor_recalibrated",
                    f"Cursor re-detected at ({pos.x}, {pos.y})",
                    agent_name="navigator",
                )
                return True
            return False

        elif action == "navigate_sidebar":
            # Try to parse recovery steps from the hypothesis result
            try:
                text = hypothesis.result or ""
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text.strip())
                recovery_steps = data.get("recovery_steps", [])
                if recovery_steps:
                    # Try clicking the first recovery target
                    target = recovery_steps[0]
                    frame = self.sensor.capture()
                    element = self.vision.find_element(
                        Image.fromarray(frame), target
                    )
                    if element:
                        return self.locator.verified_click(
                            element.center_x, element.center_y
                        )
            except (json.JSONDecodeError, IndexError):
                pass
            return False

        return False
