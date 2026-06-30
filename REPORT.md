# Assignment 4 – The Great Object Hunt

**Student Name:** Tanish Dharaie

**Roll Number:** 25B2217

**Electronics & Robotics Club, IIT Bombay**

---

# Project Objective

The objective of this assignment is to develop an autonomous mobile robot capable of locating a user-specified object inside an unknown environment, navigating toward it safely, and reporting successful mission completion.

The robot integrates computer vision, sensor fusion, state estimation, and autonomous motion control into a unified perception pipeline.

---

# System Overview

The implemented system follows the architecture below.

```
User Input

↓

YOLOv8 Detection

↓

ByteTrack Multi-Object Tracking

↓

Kalman Filter Prediction

↓

Depth + LIDAR Fusion

↓

Mission State Machine

↓

Velocity Controller

↓

Robot Motion
```

---

# Distance Estimation

Object distance is estimated using multi-sensor fusion.

## Primary Source

Depth camera measurements extracted from the lower-central region of the detected bounding box.

Using the median depth value significantly improves robustness against invalid depth pixels.

---

## Secondary Source

2D LIDAR measurements are projected toward the object's bearing.

A median filter is applied over a small angular window to suppress noise.

---

## Fusion Strategy

The final distance estimate is computed by combining

- depth measurements
- laser measurements

using conservative minimum-range fusion.

Advantages include

- improved robustness
- reduced depth noise
- better performance at close distances
- reliable stopping behavior

---

# Object Detection and Tracking

Object perception is performed using YOLOv8.

The detector provides

- object category
- confidence
- bounding box

Multiple detections are associated using ByteTrack to preserve target identity across frames.

The robot prioritizes

1. previously tracked ID

2. Kalman prediction

3. highest confidence candidate

This strategy minimizes identity switching when multiple similar objects appear.

---

# Kalman Filter

A six-state Kalman Filter estimates

- image position
- velocity
- target distance

State vector

```
[cx, cy, distance, vx, vy, vdistance]
```

The prediction model enables

- temporary occlusion handling
- smoother object following
- reduced steering oscillation
- stable approach behavior

---

# Search Strategy

When the target is not visible, the robot enters the SEARCH state.

The robot

- performs alternating rotational scans
- periodically advances into unexplored regions
- checks front LIDAR clearance
- avoids rotating indefinitely in one direction

This exploration behavior increases the probability of target acquisition while maintaining collision safety.

---

# Navigation Strategy

Navigation consists of two phases.

## Target Alignment

The robot first rotates until the object is centered inside the camera frame.

Only after alignment does forward motion begin.

---

## Object Approach

Forward speed is proportional to target distance.

Angular corrections continue during motion to compensate for heading error.

Velocity commands are acceleration-limited to eliminate sudden motion and wheel skidding.

---

# Collision Avoidance

The robot continuously monitors frontal LIDAR clearance.

Safety mechanisms include

- emergency stop
- dynamic speed reduction
- obstacle-aware search behavior
- minimum stopping distance

This prevents the robot from colliding with obstacles while approaching the target.

---

# Mission State Machine

The autonomous behavior is implemented using a finite-state machine.

```
IDLE

↓

SEARCH

↓

TRACK

↓

APPROACH

↓

RECOVER

↓

COMPLETE
```

Each state has clearly defined transition conditions, resulting in predictable and reliable autonomous behavior.

---

# Software Technologies

- ROS2 Jazzy
- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- NumPy
- CvBridge

---

# Key Contributions

- Autonomous object search
- Multi-object target association
- Kalman Filter target prediction
- Multi-sensor distance fusion
- Intelligent recovery behavior
- Smooth motion control
- Collision-aware navigation
- Continuous assistant mode
- Professional visualization dashboard

---

# Current Limitations

Although the implemented system satisfies the assignment objectives, several practical limitations remain.

- Traffic cones are not included in the standard COCO dataset and require a custom-trained detector.
- The robot performs reactive navigation and does not maintain a global map.
- Long-duration target occlusions may require additional localization support.
- Global exploration would benefit from integration with Navigation2 and SLAM.

---

# Future Work

Future improvements include

- Navigation2 integration
- Visual SLAM
- Robot Localization EKF
- NVIDIA Isaac ROS Nvblox
- Semantic mapping
- Behavior Tree navigation
- Dynamic obstacle prediction
- Custom object detection models
- Multi-camera perception

---

# Conclusion

The developed autonomous object hunter successfully integrates perception, tracking, state estimation, sensor fusion, and motion control into a complete robotic pipeline. The robot is capable of autonomously searching for user-specified objects, maintaining robust target tracking using Kalman filtering and multi-object association, safely navigating through the environment using fused depth and LIDAR information, and terminating the mission upon successfully reaching the target.

The overall architecture closely follows modern autonomous robotics design principles by separating perception, estimation, decision-making, and control into modular components, providing a scalable foundation for future integration with full navigation and SLAM frameworks.
