# The Great Object Hunt - ERC Assignment 4

**Electronics & Robotics Club, IIT Bombay**

## Project Description
Autonomous object detection, tracking, and navigation system using YOLOv8, depth camera, and LIDAR for a home assistant robot.

## Features Implemented
- Real-time target input through terminal (Continuous Assistant Mode)
- Robust object tracking using YOLOv8 + OpenCV CSRT Tracker
- Multi-sensor distance estimation (Depth + LIDAR)
- Intelligent search behavior with alternating rotation
- Smooth centering and straight-line approach
- Obstacle avoidance using LIDAR
- Safe stopping at target
- Professional real-time dashboard



**Electronics & Robotics Club, IIT Bombay**

## Project Description
Autonomous object detection, tracking, and navigation system using YOLOv8, depth camera, and LIDAR for a home assistant robot.

## Features Implemented
- Real-time target input through terminal (Continuous Assistant Mode)
- Robust object tracking using YOLOv8 + OpenCV CSRT Tracker
- Multi-sensor distance estimation (Depth + LIDAR)
- Intelligent search behavior with alternating rotation
- Smooth centering and straight-line approach
- Obstacle avoidance using LIDAR
- Safe stopping at target
- real-time dashboard

## How to Run

```bash
# Terminal 1: Start Gazebo Simulation
ros2 launch erc_gazebo_sensors spawn_robot.launch.py


# Terminal 2: Run Object Hunter
ros2 run erc_gazebo_sensors_py object_hunter
