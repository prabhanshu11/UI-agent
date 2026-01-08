from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

class ActionType(Enum):
    CLICK = auto()
    TYPE = auto()
    SCROLL = auto()
    WAIT = auto()
    NOP = auto()  # No Operation - useful for course correction ticks where no change is needed

@dataclass
class Action:
    type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UIState:
    timestamp: float
    screenshot: Optional[Any] = None  # Placeholder for image data
    dom_tree: Optional[Any] = None    # Placeholder for structure data
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Goal:
    intent: str
    target_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
