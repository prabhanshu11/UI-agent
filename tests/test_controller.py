import time
import pytest
from unittest.mock import MagicMock, patch
from core.controller import AgentController
from core.types import Goal, ActionType, UIState

@pytest.fixture
def mock_perception():
    with patch('core.controller.PerceptionSystem') as MockPerception:
        # Configure the mock instance
        instance = MockPerception.return_value
        instance.capture.return_value = UIState(timestamp=time.time())
        yield instance

def test_controller_init(mock_perception):
    controller = AgentController(frequency=10.0, max_ticks=100)
    assert controller.frequency == 10.0
    assert controller.tick_period_s == 0.1
    assert controller.max_ticks == 100
    assert controller.current_tick == 0
    # Ensure perception was initialized
    assert controller.perception is not None

def test_spin_once(mock_perception):
    controller = AgentController()
    controller.set_goal(Goal(intent="test"))
    
    controller.spin_once()
    
    assert controller.last_state is not None
    assert controller.last_action is not None
    assert controller.last_action.type == ActionType.NOP
    # Ensure capture was called
    mock_perception.capture.assert_called_once()

def test_spin_max_ticks(mock_perception):
    # Run at a high frequency for a short burst
    controller = AgentController(frequency=100.0, max_ticks=5)
    
    start_time = time.perf_counter()
    controller.spin()
    duration = time.perf_counter() - start_time
    
    assert controller.current_tick == 5
    # Should take roughly 5 ticks * 0.01s = 0.05s
    # We allow some buffer for test overhead
    assert duration >= 0.04 

def test_shutdown(mock_perception):
    controller = AgentController(frequency=10.0)
    
    # We'll use a trick to shutdown after one spin
    def mock_spin_once():
        controller.shutdown()
        
    controller.spin_once = mock_spin_once
    controller.spin()
    
    assert controller.current_tick == 0 # Shutdown before increment in this mock
    mock_perception.shutdown.assert_called_once()
