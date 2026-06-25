# Assignment 4: The Great Object Hunt - Short Report

**Student Name:** [Tanish Dharaie]  
**Roll Number:** [25B2217]  
**Date:** 25 June 2026

## 1. How Distance was Estimated
- Primary method: Median value of a 13×13 pixel region around the center of the tracked bounding box from `/camera/depth_image`.
- Fallback 1: Minimum range from `/scan` (LIDAR) in the front sector (±40 degrees).
- Fallback 2: Bounding box size heuristic (large box = very close).
- This multi-sensor fusion makes distance estimation stable even when depth camera fails at close range.

## 2. How Search Behaviour Works
- When target is not visible, the robot performs an intelligent alternating scan: rotates left and right (±60°) while moving slowly forward.
- Uses OpenCV CSRT visual tracker + YOLO majority voting for robust re-acquisition.
- On target loss, first attempts local re-acquisition (small sweep), then full search.
- This allows the robot to explore other rooms effectively.

## 3. How Robot Decides When to Move and Stop
- **Centering Phase**: Strong angular correction until object is nearly centered in the frame.
- **Approach Phase**: Once centered, moves mostly straight forward with small corrections. Speed is proportional to estimated distance.
- **Stopping Condition**: Robot stops when fused distance ≤ 0.6m OR bounding box becomes very large.
- **Obstacle Avoidance**: If LIDAR detects obstacle < 0.45m ahead, robot moves backward briefly.
- **State Machine with Hysteresis**: Prevents oscillation between searching and tracking due to temporary misdetections.

## Bonus Features Implemented
- Continuous Assistant Mode (can hunt multiple objects without restarting the node)
- Enhanced real-time dashboard showing target, status, distance, and FPS
- Robust visual tracking using CSRT tracker + temporal voting
- Multi-sensor distance fusion (Depth + LIDAR + bbox size)

**GitHub Repository:** (https://github.com/tanishdharaie-cell/erc-assignment-4-object-hunt.git) 
**Declaration:** This submission fulfills all stages (1-6) and most bonus challenges.
