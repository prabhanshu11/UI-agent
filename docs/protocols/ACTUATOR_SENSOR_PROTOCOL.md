# Actuator-Sensor Mapping Protocol (ASMP)

> Ultrathink Design Document
> Created: 2026-01-15
> Part of: Star Trek Computer / UI-Agent / LCARS ecosystem

---

## Abstract

A generalized protocol for mapping any **actuator** (thing that moves/changes state) to any **sensor** (thing that observes) through an **API layer** that enables **agent-driven knowledge acquisition and action execution**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACTUATOR-SENSOR PROTOCOL                      │
├─────────────────────────────────────────────────────────────────┤
│  ACTUATOR ←──→ API ←──→ KNOWLEDGE/ACTION ←──→ AGENT (Claude)   │
│  (PTZ cam)     (ONVIF)   (LLM Vision)          (Hooks)          │
│  (Robot arm)   (ROS)     (Object Detection)    (Commands)       │
│  (Drone)       (MAVLink) (SLAM)                (Goals)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Concept: Hybrid Binary Spatial Search

### The Problem
Searching a continuous space (camera FOV, robot workspace, drone coverage) linearly is O(n).
We need O(log n) search with real-time interruption capability.

### The Solution: 2D Binary Search + Continuous Streaming

```
Phase 1: COARSE SWEEP (Continuous, Low Quality)
├── Camera moves continuously through space
├── Video stream analyzed in real-time by LLM
├── Passive snapshots taken every N ms (motion blur acceptable)
├── Agent receives "hooks" - detection events
└── If target POSSIBLY detected → Phase 2

Phase 2: BINARY REFINEMENT (Stop-and-Capture, High Quality)
├── Stop at detection region
├── Take HIGH QUALITY snapshot (HDR, no motion blur)
├── LLM confirms target with high confidence
├── If confirmed → DONE
├── If false positive → Binary subdivide remaining space
└── Repeat until found or space exhausted

Phase 3: LOCK & TRACK (Optional)
├── Maintain position on target
├── Periodic re-confirmation
└── Alert on target movement/disappearance
```

---

## Protocol Specification

### 1. Actuator Interface

```python
class ActuatorInterface(Protocol):
    """Any device that can change position/state in a searchable space."""

    def get_position(self) -> Position:
        """Current position in actuator's coordinate space."""
        ...

    def move_continuous(self, velocity: Vector, duration: float) -> None:
        """Move at velocity for duration (for coarse sweep)."""
        ...

    def move_absolute(self, position: Position) -> None:
        """Move to exact position (for refinement)."""
        ...

    def stop(self) -> None:
        """Immediate stop (for detection interrupt)."""
        ...

    def get_bounds(self) -> Bounds:
        """Searchable space limits."""
        ...

@dataclass
class Position:
    """N-dimensional position."""
    coords: tuple[float, ...]  # (pan, tilt) or (x, y, z) or (joint1, joint2, ...)

@dataclass
class Bounds:
    """Search space limits."""
    min_pos: Position
    max_pos: Position
    dimensions: int  # 1D, 2D, 3D, etc.
```

### 2. Sensor Interface

```python
class SensorInterface(Protocol):
    """Any device that observes the space affected by actuator."""

    def capture_passive(self) -> SensorReading:
        """Quick capture, may have motion artifacts."""
        ...

    def capture_active(self) -> SensorReading:
        """High-quality capture, requires stable position."""
        ...

    def start_stream(self, callback: Callable[[Frame], None]) -> StreamHandle:
        """Start continuous stream with frame callback."""
        ...

    def stop_stream(self, handle: StreamHandle) -> None:
        """Stop continuous stream."""
        ...

@dataclass
class SensorReading:
    timestamp: float  # Unix epoch
    position: Position  # Actuator position when captured
    data: Any  # Image, point cloud, audio, etc.
    quality: Literal["passive", "active"]
```

### 3. Knowledge/Action Interface (Agent Hooks)

```python
class AgentHook(Protocol):
    """Interface for LLM agent to receive observations and issue commands."""

    async def analyze(self, reading: SensorReading, query: str) -> AnalysisResult:
        """Send reading to LLM for analysis."""
        ...

    async def on_detection(self, event: DetectionEvent) -> AgentDecision:
        """Called when potential target detected during sweep."""
        ...

@dataclass
class DetectionEvent:
    reading: SensorReading
    confidence: float
    description: str
    bounding_box: Optional[tuple[int, int, int, int]]  # For visual sensors

@dataclass
class AgentDecision:
    action: Literal["continue", "stop", "refine", "abort"]
    reason: str
    target_position: Optional[Position]  # For "refine" action
```

---

## Algorithm: Hybrid Binary Spatial Search

```python
async def spatial_search(
    actuator: ActuatorInterface,
    sensor: SensorInterface,
    agent: AgentHook,
    target_query: str,
    confidence_threshold: float = 0.7,
) -> SearchResult:
    """
    Find target in actuator's workspace using hybrid binary search.

    Time Complexity: O(log n) for n = search space resolution
    """

    bounds = actuator.get_bounds()
    search_regions = [bounds]  # Start with full space
    best_detection: Optional[DetectionEvent] = None

    while search_regions:
        region = search_regions.pop(0)

        # === PHASE 1: COARSE SWEEP ===
        detections = await coarse_sweep(
            actuator, sensor, agent, region, target_query
        )

        if not detections:
            continue  # Nothing in this region

        # === PHASE 2: BINARY REFINEMENT ===
        for detection in sorted(detections, key=lambda d: -d.confidence):
            # Stop and take high-quality capture
            actuator.move_absolute(detection.reading.position)
            await asyncio.sleep(0.5)  # Stabilize
            actuator.stop()

            hq_reading = sensor.capture_active()
            result = await agent.analyze(hq_reading, target_query)

            if result.confidence >= confidence_threshold:
                return SearchResult(
                    found=True,
                    position=detection.reading.position,
                    confirmation=hq_reading,
                    confidence=result.confidence,
                )

            # False positive - subdivide region
            if result.confidence > 0.3:  # Promising but not confirmed
                subregions = binary_subdivide(region, detection.reading.position)
                search_regions.extend(subregions)

    return SearchResult(found=False)


async def coarse_sweep(
    actuator: ActuatorInterface,
    sensor: SensorInterface,
    agent: AgentHook,
    region: Bounds,
    target_query: str,
) -> list[DetectionEvent]:
    """
    Continuous sweep through region with passive captures.
    Returns potential detections for refinement.
    """
    detections = []
    detection_queue = asyncio.Queue()

    # Start continuous stream with real-time analysis
    async def frame_callback(frame: Frame):
        # Quick analysis (fast model, lower accuracy OK)
        result = await agent.analyze(
            SensorReading(
                timestamp=time.time(),
                position=actuator.get_position(),
                data=frame,
                quality="passive"
            ),
            target_query
        )

        if result.possible_match:
            await detection_queue.put(DetectionEvent(
                reading=result.reading,
                confidence=result.confidence,
                description=result.description,
            ))

    # Start sweep motion
    stream = sensor.start_stream(frame_callback)
    actuator.move_continuous(
        velocity=calculate_sweep_velocity(region),
        duration=calculate_sweep_duration(region)
    )

    # Collect detections during sweep
    while not sweep_complete(actuator, region):
        try:
            detection = await asyncio.wait_for(
                detection_queue.get(), timeout=0.1
            )
            detections.append(detection)

            # Agent can interrupt sweep
            decision = await agent.on_detection(detection)
            if decision.action == "stop":
                actuator.stop()
                break
        except asyncio.TimeoutError:
            continue

    sensor.stop_stream(stream)
    return detections


def binary_subdivide(region: Bounds, detection_pos: Position) -> list[Bounds]:
    """
    Subdivide region into quadrants (2D) or octants (3D) centered on detection.
    Priority order: closest to detection position first.
    """
    # Implementation depends on dimensionality
    ...
```

---

## Implementation: Tapo C210 Camera

### Actuator: ONVIF PTZ Controller

```python
class TapoPTZActuator(ActuatorInterface):
    """Implements ActuatorInterface for Tapo C210 via ONVIF."""

    def __init__(self, host: str, username: str, password: str):
        self.controller = ONVIFPTZController(host, username, password)

    def get_position(self) -> Position:
        pos = self.controller.get_position()
        return Position(coords=(pos.pan, pos.tilt))

    def move_continuous(self, velocity: Vector, duration: float):
        self.controller.move_continuous(velocity[0], velocity[1], duration)

    def move_absolute(self, position: Position):
        self.controller.move_absolute(position.coords[0], position.coords[1])

    def stop(self):
        self.controller.stop()

    def get_bounds(self) -> Bounds:
        return Bounds(
            min_pos=Position((-1.0, -1.0)),  # ONVIF normalized
            max_pos=Position((1.0, 1.0)),
            dimensions=2
        )
```

### Sensor: RTSP Stream + Snapshot

```python
class TapoRTSPSensor(SensorInterface):
    """Implements SensorInterface for Tapo C210 via RTSP."""

    def __init__(self, rtsp_url: str):
        self.stream = StreamCapture(rtsp_url)

    def capture_passive(self) -> SensorReading:
        """Grab frame from stream (may have motion blur)."""
        frame = self.stream.get_frame()
        return SensorReading(
            timestamp=time.time(),
            position=None,  # Filled by caller
            data=frame,
            quality="passive"
        )

    def capture_active(self) -> SensorReading:
        """
        Wait for stream to stabilize, then capture.
        Could also trigger camera's internal HDR capture via ONVIF.
        """
        time.sleep(0.3)  # Wait for motion blur to clear
        frame = self.stream.get_frame()
        return SensorReading(
            timestamp=time.time(),
            position=None,
            data=frame,
            quality="active"
        )

    def start_stream(self, callback):
        # Return handle to async frame delivery
        ...
```

### Agent Hook: OpenRouter LLM Vision

```python
class GeminiVisionHook(AgentHook):
    """Implements AgentHook using Gemini 3 Flash via OpenRouter."""

    def __init__(self):
        self.vision = LLMVision(model="gemini-3")
        self.fast_model = "gemini-3"  # For coarse sweep
        self.accurate_model = "claude-sonnet"  # For confirmation

    async def analyze(self, reading: SensorReading, query: str) -> AnalysisResult:
        model = self.fast_model if reading.quality == "passive" else self.accurate_model
        result = self.vision.analyze_screen(
            reading.data,
            task=f"Is this visible: {query}? Reply DETECTED:0.X or NOT_FOUND."
        )
        return AnalysisResult(
            possible_match="DETECTED" in result.screen_description.upper(),
            confidence=extract_confidence(result.screen_description),
            description=result.screen_description,
        )

    async def on_detection(self, event: DetectionEvent) -> AgentDecision:
        if event.confidence > 0.8:
            return AgentDecision(action="stop", reason="High confidence detection")
        return AgentDecision(action="continue", reason="Continue sweep")
```

---

## Implementation: ESP32 Mouse + HDMI Capture (Cursor Detection)

A concrete application of ASMP for detecting cursor position on a multi-monitor
Windows system using an ESP32 BT HID mouse as the actuator and a USB HDMI
capture card as the sensor.

### Problem Statement

The ESP32 HID mouse sends relative (dx, dy) movements. On multi-monitor Windows,
the cursor may be on a different monitor than the one captured by HDMI. We need
to locate the cursor on the captured screen without absolute positioning.

### Actuator: ESP32 BT HID Mouse

```python
class ESP32MouseActuator:
    """ASMP actuator adapter for ESP32Mouse."""

    def __init__(self, mouse: ESP32Mouse):
        self.mouse = mouse

    def get_position(self) -> Position:
        x, y = self.mouse.get_position()
        return Position(coords=(float(x), float(y)))

    def move_continuous(self, velocity: Vector, duration: float):
        # Move in discrete steps simulating continuous motion
        dx = int(velocity[0] * duration)
        dy = int(velocity[1] * duration)
        self.mouse._send_chunked(dx, dy)

    def move_absolute(self, position: Position):
        # Relative mouse — uses estimated position for delta calc
        self.mouse.move_to(int(position.coords[0]), int(position.coords[1]))

    def stop(self):
        pass  # Relative mouse — no continuous motion to stop

    def get_bounds(self) -> Bounds:
        # Virtual desktop bounds (may span multiple monitors)
        return Bounds(
            min_pos=Position((0.0, 0.0)),
            max_pos=Position((5000.0, 3000.0)),  # Conservative max
            dimensions=2
        )
```

### Sensor: USB HDMI Capture Card

```python
class HDMISensor:
    """ASMP sensor adapter for USB HDMI capture via ffmpeg."""

    def __init__(self, device: str = "/dev/video0",
                 width: int = 1920, height: int = 1080):
        self.device = device
        self.width = width
        self.height = height

    def capture_passive(self) -> SensorReading:
        frame = self._capture_frame()
        return SensorReading(
            timestamp=time.time(), position=None,
            data=frame, quality="passive"
        )

    def capture_active(self) -> SensorReading:
        time.sleep(0.3)  # Wait for stability after movement
        # Use 3-frame capture for stable output
        frame = self._capture_frame(frames=3)
        return SensorReading(
            timestamp=time.time(), position=None,
            data=frame, quality="active"
        )

    def _capture_frame(self, frames: int = 1) -> np.ndarray:
        subprocess.run([
            "ffmpeg", "-y", "-f", "v4l2",
            "-input_format", "yuyv422",
            "-video_size", f"{self.width}x{self.height}",
            "-i", self.device,
            "-frames:v", str(frames), "-update", "1",
            "/tmp/hdmi_frame.png"
        ], capture_output=True)
        return np.array(Image.open("/tmp/hdmi_frame.png"))
```

### Detection Strategy: Frame Subtraction

Instead of using LLM vision (overkill for cursor detection), this uses
deterministic frame subtraction — the cursor is the only thing that changes
between frames when we control the movement:

```
Frame 1 (before move)    Frame 2 (after +50,+50)    |Frame 2 - Frame 1|
┌─────────────────────┐  ┌─────────────────────┐    ┌─────────────────┐
│                     │  │                     │    │                 │
│     ↑               │  │                     │    │     ∎ ← cursor  │
│     cursor          │  │         ↑           │    │       was here  │
│                     │  │         cursor       │    │         ∎ ← now│
│                     │  │                     │    │                 │
└─────────────────────┘  └─────────────────────┘    └─────────────────┘
```

This maps to ASMP phases:
- **Coarse sweep**: Move cursor right in 200px steps, check frame-diff at each
- **Binary refinement**: Not needed — frame-diff gives exact pixel coordinates
- **Lock & track**: Periodic re-verification via frame-diff before clicks

### Usage

```python
from src.hardware.cursor_locator import locate_cursor, verified_click
from src.hardware.esp32_mouse import ESP32Mouse

mouse = ESP32Mouse('/dev/ttyUSB0')
pos = locate_cursor(mouse)  # Sweep + frame-diff
if pos:
    print(f"Cursor found at {pos} on HDMI screen")
    verified_click(mouse, target_x, target_y)  # Move, verify, click
```

---

## Integration Points

### 1. UI-Agent Action Library

Add to `/home/prabhanshu/Programs/UI-agent/src/core/types.py`:

```python
class ActionType(Enum):
    # ... existing actions ...
    SPATIAL_SEARCH = auto()  # New: Search space for target
    PTZ_MOVE = auto()        # New: Camera pan/tilt/zoom
    PTZ_STOP = auto()        # New: Stop camera motion
```

### 2. LCARS Subsystems

Add to `/home/prabhanshu/Programs/claude-computer/src/lcars/subsystems/`:

```
subsystems/
├── __init__.py
├── spatial_search.py    # ASMP implementation
├── cameras/
│   └── tapo_c210.py     # Tapo-specific actuator+sensor
└── protocols/
    └── asmp.py          # Protocol interfaces
```

### 3. Tapo Monitor Scripts

Update `/home/prabhanshu/Programs/tapo-c210-monitor/scripts/find_desk.py` to use ASMP.

---

## Performance Characteristics

| Metric | Linear Sweep | ASMP Hybrid Binary |
|--------|--------------|-------------------|
| Time Complexity | O(n) | O(log n) |
| API Calls (worst) | 9 (3x3 grid) | 4-5 (binary refinement) |
| Detection Latency | After full sweep | Real-time interrupt |
| Quality | All low-quality | Low coarse + High confirm |
| False Positive Rate | High (keyword match) | Low (dual-stage verify) |

---

## Future Extensions

1. **3D Search** - Drone with (x, y, z) actuation
2. **Multi-Sensor Fusion** - Camera + LIDAR + Audio
3. **Learned Priors** - Start search where target usually is
4. **Continuous Tracking** - After finding, maintain lock
5. **Multi-Target** - Find all instances, not just first

---

## Usage Example

```python
# Find user's desk using Tapo camera
actuator = TapoPTZActuator("192.168.29.183", "prabhanshu", "iamapantar")
sensor = TapoRTSPSensor("rtsp://prabhanshu:iamapantar@192.168.29.183/stream1")
agent = GeminiVisionHook()

result = await spatial_search(
    actuator=actuator,
    sensor=sensor,
    agent=agent,
    target_query="computer desk with laptop or monitor screen",
    confidence_threshold=0.7,
)

if result.found:
    print(f"Desk found at position {result.position}")
    print(f"Confidence: {result.confidence}")
    # Camera is now pointing at desk
```

---

## References

- Original vision: `docs/protocols/SPATIAL_SEARCH_PROTOCOL_RAW.md`
- UI-Agent PPE architecture: `/home/prabhanshu/Programs/UI-agent/src/core/controller.py`
- LCARS subsystems: `/home/prabhanshu/Programs/claude-computer/src/lcars/subsystems/`
- Tapo ONVIF control: `/home/prabhanshu/Programs/tapo-c210-monitor/src/tapo_c210_monitor/ptz_mapper/`
