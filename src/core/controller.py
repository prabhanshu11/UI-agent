import time
import logging
from typing import Optional
from .types import UIState, Action, ActionType, Goal
from ..perception import PerceptionSystem
from .log_manager import LogManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentController:
    """
    Main controller for the UI-Agent. 
    Implements a Perception-Planning-Execution (PPE) cycle as a deterministic control loop.
    """

    def __init__(self, frequency: float = 1.0, max_ticks: Optional[int] = None):
        """
        :param frequency: Tick frequency in Hz (ticks per second).
        :param max_ticks: Optional limit on the number of ticks to run.
        """
        self.frequency = frequency
        self.tick_period_s = 1.0 / frequency if frequency > 0 else 0
        self.max_ticks = max_ticks
        self.current_tick = 0
        self._shutdown_requested = False
        
        # Subsystems
        self.perception = PerceptionSystem()
        self.log_manager = LogManager()
        
        # State
        self.current_goal: Optional[Goal] = None
        self.last_state: Optional[UIState] = None
        self.last_action: Optional[Action] = None

    def set_goal(self, goal: Goal):
        """Sets the active goal for the agent."""
        logger.info(f"Setting goal: {goal.intent}")
        self.current_goal = goal

    def spin(self):
        """Starts the main control loop at the specified frequency."""
        logger.info(f"Starting controller spin at {self.frequency}Hz")
        self._shutdown_requested = False
        
        # Cleanup old logs before starting
        stats = self.log_manager.prune_logs()
        if stats['deleted_files'] > 0:
            logger.info(f"Log cleanup: removed {stats['deleted_files']} files ({stats['freed_bytes']/1024/1024:.2f} MB)")
        
        try:
            while not self._shutdown_requested:
                if self.max_ticks is not None and self.current_tick >= self.max_ticks:
                    logger.info("Reached max_ticks. Shutting down.")
                    break
                
                start_time = time.perf_counter()
                
                self.spin_once()
                
                if self._shutdown_requested:
                    break
                
                self._sleep_for_tick(start_time)
                self.current_tick += 1
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested via KeyboardInterrupt")
        finally:
            self.shutdown()

    def spin_once(self):
        """Executes a single Perception-Planning-Execution (PPE) cycle."""
        logger.debug(f"Tick {self.current_tick} starting")
        
        # 1. Perception
        state = self._perceive()
        self.last_state = state
        
        # 2. Planning
        action = self._plan(state)
        self.last_action = action
        
        # 3. Execution
        self._execute(action)
        
        logger.debug(f"Tick {self.current_tick} complete")

    def shutdown(self):
        """Triggers graceful shutdown of the controller."""
        if self._shutdown_requested:
            logger.info("Shutdown already requested/completed.")
            return

        logger.info("Shutting down AgentController")
        self._shutdown_requested = True
        self.perception.shutdown()

    def _perceive(self) -> UIState:
        """Capture the current state of the UI environment."""
        state = self.perception.capture()
        return state

    def _plan(self, state: UIState) -> Action:
        """Decide the next action based on current state and goal."""
        if not self.current_goal:
            return Action(type=ActionType.NOP)
        
        # TODO: Implement complex planning (LLM or rule-based)
        # For now, just return NOP as a baseline course-correction primitive
        return Action(type=ActionType.NOP)

    def _execute(self, action: Action):
        """Perform the actual UI action."""
        if action.type == ActionType.NOP:
            return
            
        logger.info(f"Executing action: {action.type.name}")
        # TODO: Implement driver-based execution (PyAutoGUI, Selenium, etc.)

    def _sleep_for_tick(self, start_time: float):
        """Maintains the loop frequency by sleeping for the remainder of the tick."""
        elapsed = time.perf_counter() - start_time
        sleep_time = self.tick_period_s - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif sleep_time < 0 and self.frequency > 0:
            logger.warning(f"Tick period exceeded by {-sleep_time:.4f}s. Loop frequency degraded.")
