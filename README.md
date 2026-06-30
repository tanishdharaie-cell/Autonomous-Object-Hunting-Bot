# The Great Object Hunt – Autonomous Object Hunter
### Electronics & Robotics Club, IIT Bombay

An autonomous perception and navigation system developed for ERC Assignment 4. The robot continuously searches for a user-specified object, tracks it using computer vision, estimates its distance through multi-sensor fusion, autonomously approaches the target while avoiding collisions, and reports mission completion upon successful arrival.

---

## Overview

The system integrates modern computer vision, sensor fusion, and robot control techniques to achieve reliable autonomous object hunting inside a simulated indoor environment.

Unlike a conventional object detector, the robot continuously performs perception, target selection, tracking, motion planning, and obstacle-aware navigation using a finite-state control architecture.

The complete pipeline is designed to mimic the perception and navigation strategy used in autonomous service robots.

---

## System Architecture

```
                User Target
                     │
                     ▼
          YOLOv8 Object Detection
                     │
                     ▼
      Multi-Object Tracking (ByteTrack)
                     │
                     ▼
         Kalman Filter Prediction
                     │
                     ▼
      Target Selection & Association
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
Depth Camera                 Laser Scanner
      │                             │
      └──────────Sensor Fusion──────┘
                     │
                     ▼
        Mission State Machine
 SEARCH → TRACK → APPROACH → RECOVER
                     │
                     ▼
       Velocity Controller (cmd_vel)
                     │
                     ▼
              Differential Drive Robot
```

---

# Features

## Autonomous Mission Execution

- Terminal-based target selection
- Continuous autonomous search
- Automatic target acquisition
- Autonomous object following
- Goal completion detection
- Continuous assistant mode without restarting the node

---

## Computer Vision

- YOLOv8 object detection
- Multi-object tracking using ByteTrack
- Persistent target identity
- Class-specific confidence thresholds
- Native COCO class validation
- Real-time visual dashboard

---

## Sensor Fusion

Distance estimation combines:

- RGB Camera
- Depth Camera
- 2D LIDAR

The robot fuses multiple sensing modalities to obtain a robust estimate of object distance while reducing the influence of noisy depth measurements.

---

## Kalman Filter

A six-state Kalman Filter predicts

- object position
- object velocity
- estimated distance

during temporary detection loss.

State vector:

```
[cx, cy, distance, vx, vy, vdistance]
```

Benefits:

- smoother tracking
- reduced jitter
- target prediction during short occlusions
- stable motion commands

---

## Mission State Machine

The robot operates using five autonomous states.

### SEARCH

- Rotational scanning
- Forward exploration when path is clear
- Intelligent alternating search direction

### TRACK

- Centers target inside camera frame
- Maintains target identity
- Kalman prediction enabled

### APPROACH

- Moves toward target
- Speed proportional to distance
- Continuous heading correction
- Dynamic slowdown near obstacles

### RECOVER

Activated when target is temporarily lost.

The robot

- predicts target location
- performs local recovery
- returns to search if target cannot be reacquired

### COMPLETE

Robot stops safely and prints

```
OBJECT FOUND
```

---

# Motion Control

The robot uses

- proportional steering
- acceleration limiting
- angular velocity limiting
- obstacle-aware velocity scaling

to eliminate sudden motion and wheel skidding.

---

# Collision Prevention

The robot continuously monitors front LIDAR clearance.

Safety behaviors include

- emergency stop
- reduced speed near obstacles
- obstacle-aware searching
- safe stopping distance

---

# Dashboard

Real-time visualization displays

- detected objects
- tracked target
- target ID
- mission state
- estimated distance
- commanded linear velocity
- commanded angular velocity
- Kalman prediction

---

# Software Stack

- ROS2 Jazzy
- Python 3.12
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- NumPy
- CvBridge

---

# Package Structure

```
erc_ws/
│
├── erc_gazebo_sensors/
│
├── erc_gazebo_sensors_py/
│      ├── object_hunter_node.py
│      ├── yolo_detection_node.py
│      └── ...
│
└── launch/
```

---

# Running the Project

## Terminal 1

Launch Gazebo

```bash
source /opt/ros/jazzy/setup.bash
source ~/erc_ws/install/setup.bash

ros2 launch erc_gazebo_sensors spawn_robot.launch.py
```

---

## Terminal 2

Run the Object Hunter

```bash
source /opt/ros/jazzy/setup.bash
source ~/erc_ws/install/setup.bash

ros2 run erc_gazebo_sensors_py object_hunter
```

---

# Supported Objects

Current YOLO model supports

- Person
- Chair
- Refrigerator

Traffic cones require a custom-trained detector since they are not part of the COCO dataset.

---

# Future Improvements

- Nav2 integration
- Visual SLAM
- Semantic mapping
- Active exploration
- Dynamic path planning
- Robot Localization EKF
- Nvblox obstacle mapping
- Multi-camera perception
- Custom object detection model
- Behavior Tree mission planner

---

# References

- Ultralytics YOLOv8
- ByteTrack (ECCV 2022)
- ROS2 Navigation2
- NVIDIA Isaac ROS
- robot_localization
- OpenCV
