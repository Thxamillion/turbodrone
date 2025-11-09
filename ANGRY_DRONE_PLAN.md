# Angry Productivity Drone - Implementation Plan

## 🎯 Project Goal
Build a drone that monitors your desk productivity. When you leave your desk, it gets progressively angrier, eventually following and "attacking" you to force you back to work.

## 📋 High-Level Phases

### Phase 1: Prototype (1-2 days)
**Goal:** Quick implementation to test core concepts

- Use Option A (Tap Frame Pump) for rapid prototyping
- Implement basic person detection using YOLO or MediaPipe
- Verify detection works with drone video feed
- Test basic anger behaviors (wobbling, movement)
- Quick and dirty - just get it working!

### Phase 2: Refactor (1 day)
**Goal:** Clean architecture for production quality

- Move to Option B (Parallel Consumer) architecture
- Create `ProductivityMonitor` class as separate module
- Implement proper state machine for anger escalation
- Add error handling and safety features
- Clean separation of concerns

### Phase 3: Polish (Ongoing)
**Goal:** Make it awesome and safe

- Add emotion detection
- Implement posture tracking
- Tune anger escalation timing
- Add LED/audio feedback if drone supports it
- Safety features: boundaries, emergency stop, distance limits

---

## 🧪 Testing Plan (Before Drone Arrives)

### Day 0: TODAY - Setup & Testing Without Drone

#### Test 1: Standalone Person Detection (15 min)
**Goal:** Verify person detection works on your webcam

```bash
python test_person_detection.py
```

**What it tests:**
- YOLO/MediaPipe detection accuracy
- Detection speed (must be <50ms per frame)
- Confidence thresholds
- Bounding box visualization

**Success criteria:**
- ✅ Detects you sitting at desk
- ✅ Detects when you leave frame
- ✅ Runs at >20 FPS

---

#### Test 2: Mock Drone Simulator (30 min)
**Goal:** Simulate drone video feed from webcam

```bash
python mock_drone_video.py
```

**What it tests:**
- Sending video over UDP to localhost:8888
- S2x packet format (header + payload)
- Frame slicing and reassembly
- Integration with existing turbodrone backend

**Success criteria:**
- ✅ Turbodrone web UI shows webcam feed
- ✅ Frame reassembly works correctly
- ✅ Video is smooth (30 FPS)

---

#### Test 3: Mock Flight Controller (30 min)
**Goal:** Test anger logic without flying

```bash
python test_anger_state_machine.py
```

**What it tests:**
- State transitions (WORKING → AWAY → ANGRY → HUNTING → ATTACKING)
- Timing (escalation intervals)
- Command generation (what movements for each state)
- Logging (see what would be sent to drone)

**Success criteria:**
- ✅ States transition correctly based on time away
- ✅ Commands logged show expected behavior
- ✅ Can reset when person returns to desk

---

#### Test 4: Full Integration with Mocks (1 hour)
**Goal:** End-to-end test with fake drone

**Components:**
- Mock video source (webcam)
- Person detection (real)
- State machine (real)
- Flight controller (mocked - logs only)

```bash
# Terminal 1: Mock drone video
python mock_drone_video.py

# Terminal 2: Turbodrone backend with detection
uvicorn web_server:app

# Terminal 3: Frontend
cd frontend && npm run dev
```

**Test scenarios:**
1. Sit at desk for 30s → drone should hover calmly
2. Leave desk for 30s → drone should start wobbling
3. Leave desk for 60s → drone should enter HUNTING mode
4. Leave desk for 90s → drone should enter ATTACKING mode
5. Return to desk → drone should calm down

**Success criteria:**
- ✅ Detection tracks your presence accurately
- ✅ State machine transitions at correct times
- ✅ Logged commands show expected behavior
- ✅ System is stable (no crashes)

---

### Day 1: DRONE ARRIVES - Real Hardware Testing

#### Test 5: Record Real Drone Packets (10 min)
**Goal:** Capture real drone video for future testing

```bash
# Enable packet dumping in web_server.py
receiver = VideoReceiverService(
    dump_packets=True,
    dump_frames=True,
    dump_dir="drone_recording_$(date +%Y%m%d)"
)
```

**Fly drone for 30 seconds in safe area**

**Success criteria:**
- ✅ Packets saved to dump directory
- ✅ Frames saved as individual JPEGs
- ✅ Can replay packets later for testing

---

#### Test 6: Person Detection with Real Drone (30 min)
**Goal:** Verify detection works with drone camera

**Test in safe area (no control, detection only):**
- Hover drone at eye level
- Stand in front of camera
- Move in and out of frame
- Check detection accuracy

**Success criteria:**
- ✅ Detects person at 2-5 meters distance
- ✅ Minimal false positives/negatives
- ✅ Detection is fast enough (video stays smooth)

---

#### Test 7: Safe Flight Tests (1 hour)
**Goal:** Test anger behaviors with safety limits

**Safety setup:**
- Fly in open area (park, backyard)
- Prop guards installed
- Observer present
- Emergency stop ready

**Progressive tests:**
1. **Calm hover:** Stand at desk, drone hovers normally
2. **Gentle wobble:** Leave desk, drone wobbles in place
3. **Slow approach:** Stay away, drone moves forward slowly
4. **Chase mode:** Keep moving, drone follows

**Success criteria:**
- ✅ All behaviors work as expected
- ✅ Emergency stop works instantly
- ✅ Drone respects safety boundaries
- ✅ No crashes or dangerous behavior

---

## 🏗️ Architecture Options

### Option A: Tap Frame Pump (Phase 1 - Prototype)

**Implementation:**
```python
# Modify web_server.py::_frame_pump_worker()
def _frame_pump_worker(src_q, stop_evt, loop):
    while not stop_evt.is_set():
        frame = src_q.get(timeout=0.5)

        # 🆕 ADD: Person detection + anger logic
        person_detected = detect_person(frame.data)
        update_productivity_state(person_detected, flight_controller)

        # Continue normal video processing
        if frame.format == "jpeg":
            jpg_bytes = frame.data
        # ...
```

**Pros:**
- ✅ Simple implementation (~10 lines)
- ✅ Fast to prototype
- ✅ Easy to debug

**Cons:**
- ❌ Blocks video display during detection
- ❌ Tight coupling
- ❌ ~19 FPS instead of 30 FPS

---

### Option B: Parallel Consumer (Phase 2 - Production)

**Implementation:**
```python
# New file: productivity_monitor.py
class ProductivityMonitor:
    def __init__(self, frame_queue, flight_controller):
        self.frame_queue = frame_queue
        self.flight_controller = flight_controller
        self.state = "WORKING"
        self.time_away = 0

    def start(self):
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def _monitor_loop(self):
        while self.running:
            frame = self.frame_queue.get(timeout=0.5)
            person_detected = self.detect_person(frame.data)
            self.update_anger_state(person_detected)

    def update_anger_state(self, person_detected):
        # State machine logic
        if not person_detected:
            self.time_away += 0.033  # 30 FPS

            if self.time_away > 90:
                self.state = "ATTACKING"
                self.flight_controller.set_axes(0.3, 0, 1.0, 0)  # Chase!
            elif self.time_away > 60:
                self.state = "HUNTING"
                self.flight_controller.set_axes(0.2, 0.1, 0.5, 0)
            elif self.time_away > 30:
                self.state = "ANGRY"
                # Wobble in place
                wobble = math.sin(time.time() * 2) * 0.3
                self.flight_controller.set_axes(0, wobble, 0, 0)
        else:
            self.time_away = 0
            self.state = "WORKING"
            self.flight_controller.set_axes(0, 0, 0, 0)  # Calm
```

**Pros:**
- ✅ 30 FPS video maintained
- ✅ Clean separation of concerns
- ✅ Easy to test independently
- ✅ Can add more monitors later
- ✅ Errors don't crash video

**Cons:**
- ❌ More complex (~100 lines)
- ❌ Threading complexity
- ❌ Need to manage lifecycle

---

## 📊 Anger State Machine

```
┌──────────────┐
│   WORKING    │  Person at desk
│  (Hovering)  │  Calm, stable hover
└──────┬───────┘
       │ Person leaves desk
       ↓
┌──────────────┐
│     AWAY     │  0-30 seconds
│  (Watching)  │  Still hovering, maybe small movements
└──────┬───────┘
       │ 30+ seconds away
       ↓
┌──────────────┐
│    ANGRY     │  30-60 seconds
│  (Wobbling)  │  Aggressive yaw movements, spinning
└──────┬───────┘
       │ 60+ seconds away
       ↓
┌──────────────┐
│   HUNTING    │  60-90 seconds
│  (Following) │  Starts moving toward you, tracking
└──────┬───────┘
       │ 90+ seconds away
       ↓
┌──────────────┐
│  ATTACKING   │  90+ seconds
│  (Charging)  │  Full speed approach, evasive patterns
└──────────────┘
       │
       │ Person returns to desk
       ↓
    (Reset to WORKING)
```

---

## 🛠️ Implementation Details

## 🎯 Detection Strategy: Progressive Implementation

The key challenge is differentiating between **"at desk"** vs **"away from desk"** and **"working"** vs **"slacking"**. We'll build this progressively:

### Phase 1a: Simple Presence Detection (Weekend - Quick Start)
**Goal:** Detect if person is in frame (binary: present or absent)

```python
import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_person_simple(jpeg_bytes):
    """Basic detection: Is anyone in the frame?"""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        return "PRESENT"
    else:
        return "ABSENT"
```

**Pros:** ✅ Very simple, ✅ Fast (20ms), ✅ Get prototype working TODAY
**Cons:** ❌ Can't tell if working or slacking, ❌ Standing up = "absent"

---

### Phase 1b: Position-Based Detection (Weekend - Better)
**Goal:** Detect if person is in the "desk zone" of the frame

```python
def is_at_desk(jpeg_bytes):
    """Detect if person is centered in frame (at desk)"""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return "ABSENT"

    # Get nose position (head position)
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    # Check if centered in frame (desk zone)
    # Center 40% of frame = at desk
    at_desk = 0.3 < nose.x < 0.7

    if at_desk:
        return "AT_DESK"
    else:
        return "AWAY"  # In frame but at edge, probably leaving
```

**Visual:**
```
Camera View:
┌─────────────────────────────────────────┐
│ [AWAY]      [AT_DESK]        [AWAY]    │
│                                         │
│               🧑                        │ ← Centered = at desk
│              /│\                        │
│              / \                        │
│         ┌──────────┐                    │
│         │   Desk   │                    │
└─────────────────────────────────────────┘
```

**Pros:** ✅ Allows brief standing/stretching, ✅ More accurate
**Cons:** ❌ Still can't detect if actually working

---

### Phase 1c: Posture Analysis (Week 1 - Activity Detection)
**Goal:** Detect WORKING posture vs SLACKING posture

```python
def analyze_work_posture(jpeg_bytes):
    """Advanced: Detect if you're in a working posture"""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return "ABSENT"

    landmarks = results.pose_landmarks.landmark

    # Get key body points
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Check if at desk (centered)
    at_desk = 0.3 < nose.x < 0.7
    if not at_desk:
        return "ABSENT"

    # Check working indicators
    # 1. Hands elevated (typing position)
    hands_up = (left_wrist.y < left_shoulder.y and
                right_wrist.y < right_shoulder.y)

    # 2. Leaning forward (engaged with computer)
    shoulder_avg_z = (left_shoulder.z + right_shoulder.z) / 2
    leaning_forward = nose.z < shoulder_avg_z

    # 3. Hands in front of body (using keyboard/mouse)
    hands_forward = (left_wrist.z < left_shoulder.z and
                    right_wrist.z < right_shoulder.z)

    # Combine signals
    working_indicators = sum([hands_up, leaning_forward, hands_forward])

    if working_indicators >= 2:
        return "WORKING"  # High confidence you're working
    elif working_indicators == 1:
        return "MAYBE_WORKING"  # Some signals
    else:
        return "SLACKING"  # Present but not working
```

**What it detects:**
- ✅ Hands up + leaning forward = typing/working
- ✅ Leaning back + hands down = slacking/on phone
- ✅ Reading (leaning forward but hands down) = maybe working

**Pros:** ✅ Detects actual work activity, ✅ No extra models needed
**Cons:** ❌ May not catch reading/thinking

---

### Phase 2: Motion Detection (Week 1 - Add Activity)
**Goal:** Detect movement (typing, mouse usage)

```python
class MotionDetector:
    def __init__(self):
        self.prev_frame = None
        self.motion_threshold = 1000  # Pixels that changed

    def detect_motion(self, frame):
        """Detect if there's movement in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return 0  # First frame

        # Calculate difference between frames
        frame_diff = cv2.absdiff(gray, self.prev_frame)

        # Count pixels that changed significantly
        motion_pixels = np.sum(frame_diff > 30)

        self.prev_frame = gray

        return motion_pixels

class EnhancedDetector:
    def __init__(self):
        self.motion_detector = MotionDetector()

    def analyze_productivity(self, jpeg_bytes):
        """Combine posture + motion for better detection"""
        # Get posture analysis
        posture_state = analyze_work_posture(jpeg_bytes)

        # Get motion
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        motion = self.motion_detector.detect_motion(frame)

        # Combine signals
        if posture_state == "ABSENT":
            return "ABSENT"

        # High motion = definitely working (typing, etc.)
        if motion > 2000:
            return "WORKING"

        # Use posture if motion is ambiguous
        if posture_state == "WORKING" and motion > 500:
            return "DEFINITELY_WORKING"
        elif posture_state == "SLACKING" and motion < 500:
            return "DEFINITELY_SLACKING"
        else:
            return "MAYBE_WORKING"
```

**Pros:** ✅ Detects typing/mouse activity, ✅ Very fast
**Cons:** ❌ Reading = no motion = "slacking"?

---

### Phase 3: Object Detection with YOLO (Week 2 - Context)
**Goal:** Detect keyboard, laptop, phone for context

```python
from ultralytics import YOLO

class AdvancedDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()
        self.yolo = YOLO('yolov8n.pt')  # Nano = fastest
        self.motion_detector = MotionDetector()
        self.frame_count = 0
        self.last_objects = {}

    def analyze_productivity(self, jpeg_bytes):
        """Multi-signal productivity detection"""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Signal 1: Posture
        posture = analyze_work_posture(jpeg_bytes)

        # Signal 2: Motion
        motion = self.motion_detector.detect_motion(frame)

        # Signal 3: Objects (run every 5th frame to save CPU)
        if self.frame_count % 5 == 0:
            self.last_objects = self.detect_objects(frame)
        self.frame_count += 1

        # Analyze context
        has_keyboard = 'keyboard' in self.last_objects
        has_laptop = 'laptop' in self.last_objects
        has_phone = 'phone' in self.last_objects

        # Decision logic
        if posture == "ABSENT":
            return "ABSENT"

        # Phone but no computer = slacking
        if has_phone and not (has_keyboard or has_laptop):
            return "SLACKING"

        # At computer with good posture + motion = working
        if (has_keyboard or has_laptop) and posture == "WORKING" and motion > 500:
            return "DEFINITELY_WORKING"

        # At computer but no motion = maybe reading
        if (has_keyboard or has_laptop) and motion < 200:
            return "MAYBE_WORKING"

        # Default to posture analysis
        return posture

    def detect_objects(self, frame):
        """Detect keyboard, laptop, phone"""
        results = self.yolo(frame, classes=[66, 67, 73])  # keyboard, mouse, laptop

        objects = {}
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 66: objects['keyboard'] = box
                elif cls_id == 67: objects['mouse'] = box
                elif cls_id == 73: objects['laptop'] = box

        return objects
```

**Pros:** ✅ Context-aware, ✅ Detects phone usage
**Cons:** ❌ Slower (50ms), ❌ More complex

---

### Phase 4 (Future): EdgeTAM Tracking (Optional Polish)
**Goal:** Smooth tracking for chase mode

**When to use EdgeTAM:**
- If you want super smooth person tracking across frames
- If you want to maintain identity when you move around
- If you want precise distance estimation for safety

**How it works:**
```python
from edgetam import EdgeTAM  # Hypothetical

class EdgeTAMDetector:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt')
        self.edgetam = EdgeTAM()
        self.initialized = False

    def first_frame(self, frame):
        """Use YOLO to find objects, then track with EdgeTAM"""
        detections = self.yolo(frame)

        # Convert YOLO bboxes to EdgeTAM prompts
        prompts = []
        for det in detections:
            center_x = (det.x1 + det.x2) / 2
            center_y = (det.y1 + det.y2) / 2
            prompts.append((center_x, center_y))

        # EdgeTAM creates precise masks
        self.masks = self.edgetam.segment(frame, prompts)
        self.initialized = True

    def track_frame(self, frame):
        """Use EdgeTAM for smooth tracking"""
        if not self.initialized:
            self.first_frame(frame)
            return

        # EdgeTAM maintains object identity across frames
        self.masks = self.edgetam.track(frame, self.masks)

        # Analyze mask positions
        person_mask = self.masks.get('person')
        keyboard_mask = self.masks.get('keyboard')

        # Calculate overlap, proximity, etc.
        return self.analyze_masks(person_mask, keyboard_mask)
```

**Pros:** ✅ Smooth tracking, ✅ Precise segmentation, ✅ Good for chase mode
**Cons:** ❌ Complex setup, ❌ Needs prompts, ❌ Slower (62ms)

---

## 📊 Detection Method Comparison

| Method | Speed | Accuracy | Complexity | Use Case |
|--------|-------|----------|------------|----------|
| **Simple Presence** | ⚡⚡⚡ 20ms | ⭐⭐ Basic | ⭐ Very Easy | Phase 1a - Quick prototype |
| **Position-Based** | ⚡⚡⚡ 20ms | ⭐⭐⭐ Good | ⭐⭐ Easy | Phase 1b - Better presence |
| **Posture Analysis** | ⚡⚡⚡ 20ms | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Medium | Phase 1c - Detect work posture |
| **Motion Detection** | ⚡⚡⚡ 5ms | ⭐⭐⭐ Good | ⭐⭐ Easy | Phase 2 - Detect activity |
| **YOLO Objects** | ⚡⚡ 50ms | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Hard | Phase 3 - Context awareness |
| **EdgeTAM Tracking** | ⚡ 62ms | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Very Hard | Phase 4 - Smooth tracking |

---

## 🎮 Enhanced State Machine with Multi-Level Detection

```python
class SmartAngerStateMachine:
    def __init__(self):
        self.state = "CALM"
        self.time_slacking = 0
        self.time_absent = 0

    def update(self, productivity_status, dt):
        """
        productivity_status values:
        - "DEFINITELY_WORKING" - Clear working indicators
        - "WORKING" - Likely working
        - "MAYBE_WORKING" - Ambiguous
        - "SLACKING" - At desk but clearly not working
        - "AWAY" - Leaving desk zone
        - "ABSENT" - Not in frame
        """

        if productivity_status in ["DEFINITELY_WORKING", "WORKING"]:
            # Reset everything - you're being productive!
            self.time_slacking = 0
            self.time_absent = 0
            self.state = "CALM"
            return {"behavior": "HOVER_CALMLY", "throttle": 0, "yaw": 0, "pitch": 0, "roll": 0}

        elif productivity_status == "MAYBE_WORKING":
            # Give benefit of the doubt, but watch closely
            self.time_slacking += dt * 0.3  # Count at reduced rate
            if self.time_slacking > 60:
                self.state = "SUSPICIOUS"
                return {"behavior": "GENTLE_WOBBLE", "throttle": 0, "yaw": 0.2, "pitch": 0, "roll": 0}
            return {"behavior": "HOVER_CALMLY", "throttle": 0, "yaw": 0, "pitch": 0, "roll": 0}

        elif productivity_status == "SLACKING":
            # You're clearly not working!
            self.time_slacking += dt

            if self.time_slacking > 120:  # 2 minutes of slacking
                self.state = "VERY_ANGRY"
                wobble = math.sin(time.time() * 5) * 0.7  # Fast aggressive wobble
                return {"behavior": "AGGRESSIVE_WOBBLE", "throttle": 0, "yaw": wobble, "pitch": 0, "roll": 0}
            elif self.time_slacking > 60:  # 1 minute
                self.state = "ANGRY"
                wobble = math.sin(time.time() * 3) * 0.4
                return {"behavior": "WOBBLE", "throttle": 0, "yaw": wobble, "pitch": 0, "roll": 0}
            elif self.time_slacking > 30:  # 30 seconds
                self.state = "ANNOYED"
                wobble = math.sin(time.time() * 2) * 0.2
                return {"behavior": "SMALL_WOBBLE", "throttle": 0, "yaw": wobble, "pitch": 0, "roll": 0}

        elif productivity_status == "AWAY":
            # You're getting up!
            self.time_absent += dt

            if self.time_absent > 30:
                self.state = "PURSUIT"
                return {"behavior": "FOLLOW", "throttle": 0, "yaw": 0.3, "pitch": 0.3, "roll": 0}
            else:
                self.state = "WATCHING"
                return {"behavior": "ROTATE_TO_TRACK", "throttle": 0, "yaw": 0.2, "pitch": 0, "roll": 0}

        elif productivity_status == "ABSENT":
            # You're gone!
            self.time_absent += dt

            if self.time_absent > 90:  # 1.5 minutes
                self.state = "ATTACK_MODE"
                return {"behavior": "CHASE", "throttle": 0.2, "yaw": 0.1, "pitch": 1.0, "roll": 0}
            elif self.time_absent > 60:  # 1 minute
                self.state = "HUNTING"
                return {"behavior": "SEARCH", "throttle": 0, "yaw": 0.5, "pitch": 0.5, "roll": 0}
            elif self.time_absent > 30:  # 30 seconds
                self.state = "CONCERNED"
                return {"behavior": "LOOK_AROUND", "throttle": 0, "yaw": 0.3, "pitch": 0, "roll": 0}

        return {"behavior": "HOVER_CALMLY", "throttle": 0, "yaw": 0, "pitch": 0, "roll": 0}
```

---

## 🚀 Recommended Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────┐
│ WEEKEND (This Week)                                         │
├─────────────────────────────────────────────────────────────┤
│ ✅ Phase 1a: Simple presence detection (MediaPipe)         │
│    - Test with webcam                                       │
│    - Get basic "present" vs "absent" working               │
│                                                             │
│ ✅ Phase 1b: Position-based detection                      │
│    - Add "at desk" zone checking                           │
│    - Test with mock drone video                            │
│                                                             │
│ ✅ Phase 1c: Posture analysis                              │
│    - Detect working vs slacking posture                    │
│    - Integrate with state machine                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ WEEK 1 (Drone Arrives)                                      │
├─────────────────────────────────────────────────────────────┤
│ ✅ Test posture detection with real drone                  │
│ ✅ Phase 2: Add motion detection                           │
│    - Detect typing/mouse activity                          │
│    - Combine with posture                                  │
│ ✅ Tune thresholds for drone camera angle                  │
│ ✅ Safe flight tests                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ WEEK 2 (Enhancement)                                        │
├─────────────────────────────────────────────────────────────┤
│ ✅ Phase 3: Add YOLO object detection                      │
│    - Detect keyboard, laptop, phone                        │
│    - Run every 5th frame for performance                   │
│    - Better "working" vs "slacking" logic                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ FUTURE (If Needed)                                          │
├─────────────────────────────────────────────────────────────┤
│ ⭕ Phase 4: EdgeTAM tracking (optional)                    │
│    - Smooth person tracking for chase mode                 │
│    - Precise distance estimation                           │
│    - Occlusion handling                                    │
└─────────────────────────────────────────────────────────────┘
```

---

### Flight Commands Reference

```python
# Hover in place
flight_controller.set_axes(
    throttle=0,  # No up/down
    yaw=0,       # No rotation
    pitch=0,     # No forward/back
    roll=0       # No left/right
)

# Wobble (angry)
wobble = math.sin(time.time() * 2) * 0.5  # -0.5 to 0.5
flight_controller.set_axes(0, wobble, 0, 0)

# Move forward slowly
flight_controller.set_axes(0, 0, 0.3, 0)

# Chase (aggressive)
flight_controller.set_axes(
    throttle=0.2,   # Slight climb
    yaw=0.1,        # Slight rotation for menacing effect
    pitch=1.0,      # Full forward
    roll=0
)

# Circle around (intimidating)
flight_controller.set_axes(0, 0.5, 0.3, 0.3)
```

---

## 🔒 Safety Features (CRITICAL!)

### 1. Distance Limits
```python
MAX_DISTANCE_FROM_DESK = 5.0  # meters
EMERGENCY_STOP_DISTANCE = 0.5  # meters from person

if distance_from_person < EMERGENCY_STOP_DISTANCE:
    flight_controller.model.land()  # Emergency land
```

### 2. Timeout
```python
MAX_HUNT_TIME = 120  # seconds
if self.time_in_hunting_mode > MAX_HUNT_TIME:
    self.return_to_base()
```

### 3. Boundary Zone
```python
# Don't follow beyond certain room
ALLOWED_AREA = BoundingBox(x1, y1, x2, y2)
if drone_position not in ALLOWED_AREA:
    self.return_to_base()
```

### 4. Manual Override
```python
# Web UI can always override
if web_ui_has_control:
    # Disable angry behavior
    monitor.pause()
```

### 5. Prop Guards
- Physical guards must be installed
- Protect people and drone
- Required for all testing

---

## 📦 Dependencies

### Python Packages
```bash
pip install opencv-python
pip install mediapipe  # OR
pip install ultralytics  # for YOLO
pip install numpy
```

### Hardware
- ✅ Karuisrc K417 drone (~$50)
- ✅ Prop guards (included with drone)
- ✅ WiFi dongle (optional but recommended)
- ✅ Webcam (for testing before drone arrives)

---

## 📝 File Structure

```
turbodrone/
├── ANGRY_DRONE_PLAN.md                    # This file
├── backend/
│   ├── web_server.py                      # Modified for Phase 1
│   ├── productivity_monitor.py            # New for Phase 2
│   ├── person_detection.py                # Detection logic
│   └── anger_state_machine.py             # State management
├── testing/
│   ├── test_person_detection.py           # Standalone detection test
│   ├── mock_drone_video.py                # Webcam → UDP simulator
│   ├── test_anger_state_machine.py        # State machine unit tests
│   └── replay_packets.py                  # Replay recorded drone packets
└── recordings/
    └── drone_recording_YYYYMMDD/          # Real drone packet dumps
```

---

## 🚀 Getting Started

### Step 1: Test Person Detection (Right Now!)
```bash
cd turbodrone/testing
python test_person_detection.py
```

### Step 2: Test with Mock Drone
```bash
# Terminal 1: Start mock drone
python mock_drone_video.py

# Terminal 2: Start backend
cd ../backend
uvicorn web_server:app

# Terminal 3: Start frontend
cd ../frontend
npm run dev

# Open: http://localhost:5173
```

### Step 3: When Drone Arrives
```bash
# Connect to drone WiFi: K417-XXXXXX
# Start backend with real drone
cd backend
echo "DRONE_TYPE=s2x" > .env
uvicorn web_server:app
```

---

## 🎓 Learning Resources

- **Turbodrone codebase walkthrough:** See conversation history
- **S2x protocol docs:** `docs/research/S2x.md`
- **MediaPipe docs:** https://google.github.io/mediapipe/
- **YOLOv8 docs:** https://docs.ultralytics.com/

---

## ✅ Success Metrics

**Phase 1 Complete When:**
- ✅ Person detection works reliably
- ✅ Video feed shows webcam/drone
- ✅ Basic anger behaviors implemented
- ✅ Can see state changes in logs

**Phase 2 Complete When:**
- ✅ ProductivityMonitor is separate class
- ✅ Video maintains 30 FPS during detection
- ✅ State machine handles all transitions
- ✅ Error handling prevents crashes
- ✅ Can toggle monitoring on/off

**Project Complete When:**
- ✅ Drone successfully "gets angry" when you leave desk
- ✅ Follows you around room
- ✅ Returns to calm when you sit down
- ✅ All safety features working
- ✅ Fun to show friends! 🎉

---

## 🐛 Troubleshooting

### Video feed not showing
- Check WiFi connection to drone
- Verify port 8888 is open
- Check console for errors

### Person detection too slow
- Switch to MediaPipe (faster than YOLO)
- Skip every other frame
- Lower video resolution

### Drone not responding to commands
- Check port 8080 is open
- Verify flight_controller is running
- Check packet logs for transmission

### State machine not transitioning
- Check timestamps in logs
- Verify person detection is working
- Print state in console

---

## 🎯 Next Steps After This Plan

1. **Implement Phase 1** (Start today with mocks)
2. **Test with real drone** (Tomorrow when it arrives)
3. **Refactor to Phase 2** (Clean architecture)
4. **Add cool features:**
   - Emotion detection (happy = calm, frustrated = more aggressive)
   - Posture checking (slouching = gentle reminder)
   - Productivity stats dashboard
   - Voice commands
   - LED light shows for different moods

---

**Let's build this! 🚁💪**
