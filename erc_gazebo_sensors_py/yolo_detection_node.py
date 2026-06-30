import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time
from collections import deque
from ultralytics import YOLO

class ObjectHunterNode(Node):
    def __init__(self):
        super().__init__('object_hunter')
        self.model = YOLO("yolov8m.pt")
        self.get_logger().info("YOLOv8m loaded - Professional Mode")

        self.bridge = CvBridge()

        self.create_subscription(Image, 'camera/image', self.image_callback, 10)
        self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_scan = None
        self.frame_lock = threading.Lock()

        self.target_class = None
        self.mission_active = False
        self.stop_distance = 0.55

        # Tracking
        self.tracker = None
        self.tracked_bbox = None
        self.detection_history = deque(maxlen=25)
        self.lost_frames = 0

        # Kalman Filter for smoothing
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        self.running = True

        self.spin_thread = threading.Thread(target=self.spin_thread_func, daemon=True)
        self.spin_thread.start()

        self.input_thread = threading.Thread(target=self.input_thread_func, daemon=True)
        self.input_thread.start()

        self.get_logger().info("Autonomous Hunter initialized with Kalman Filter and robust tracking.")

    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def input_thread_func(self):
        while rclpy.ok() and self.running:
            if not self.mission_active:
                print("\n" + "="*70)
                target = input("Enter target object: ").strip().lower()
                if target:
                    self.target_class = target
                    self.reset_state()
                    self.mission_active = True
                    print(f"🎯 Searching for: {target}")
            time.sleep(0.4)

    def reset_state(self):
        self.tracker = None
        self.tracked_bbox = None
        self.detection_history.clear()
        self.lost_frames = 0

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.frame_lock:
                self.latest_rgb = frame
        except:
            pass

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            with self.frame_lock:
                self.latest_depth = depth
        except:
            pass

    def scan_callback(self, msg):
        with self.frame_lock:
            self.latest_scan = msg

    def is_target_match(self, class_name):
        mapping = {
            "chair": ["chair"],
            "fridge": ["refrigerator", "fridge"],
            "bottle": ["bottle"],
            "person": ["person"],
            "cone": ["cone", "traffic light"],
        }
        allowed = mapping.get(self.target_class, [self.target_class])
        return any(word in class_name for word in allowed)

    def process_frame(self, rgb, depth, scan):
        h, w = rgb.shape[:2]
        results = self.model(rgb, conf=0.28, imgsz=640, verbose=False)

        target_detected = False
        bbox = None
        distance = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.model.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])

                if self.is_target_match(class_name) and conf > 0.35:
                    target_detected = True
                    bbox = (x1, y1, x2, y2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    if depth is not None:
                        distance = self.estimate_distance(depth, cx, cy)
                    self.detection_history.append(class_name)

        # Tracker management
        if target_detected and bbox and self.tracker is None:
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(rgb, bbox)
            self.tracked_bbox = bbox

        tracked_ok = False
        if self.tracker is not None:
            ok, tbox = self.tracker.update(rgb)
            if ok:
                tracked_ok = True
                x, y, ww, hh = map(int, tbox)
                self.tracked_bbox = (x, y, x+ww, y+hh)
                cx = x + ww//2
                cy = y + hh//2
                if depth is not None:
                    distance = self.estimate_distance(depth, cx, cy)
            else:
                self.lost_frames += 1

        # Control Logic
        twist = Twist()
        status = "SEARCHING"

        if self.mission_active and self.target_class:
            if self.is_obstacle_ahead(scan):
                twist.linear.x = -0.25
                status = "OBSTACLE"
            elif tracked_ok and self.tracked_bbox:
                status = "TRACKING"
                x1, y1, x2, y2 = self.tracked_bbox
                cx = (x1 + x2) // 2
                error = (w // 2 - cx) / (w / 2.0)

                if abs(error) > 0.12:
                    twist.angular.z = 1.0 * error
                    twist.linear.x = 0.0
                else:
                    if distance and distance <= self.stop_distance:
                        status = "COMPLETED"
                        self.mission_active = False
                        self.reset_state()
                        twist.linear.x = twist.angular.z = 0.0
                        print(f"\n🎉 MISSION COMPLETED! Reached the {self.target_class}")
                    else:
                        twist.linear.x = max(0.12, min(0.35, (distance or 2.0) * 0.18))
                        twist.angular.z = 0.5 * error
            else:
                status = "SEARCHING"
                self.lost_frames += 1
                self.search_direction *= -1 if self.lost_frames % 40 == 0 else 1
                twist.angular.z = 0.6 * self.search_direction
                twist.linear.x = 0.1

        self.cmd_vel_pub.publish(twist)

        # Dashboard
        dash = np.zeros((h, 450, 3), dtype=np.uint8)
        cv2.putText(dash, "AUTONOMOUS HUNTER", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.putText(dash, f"TARGET: {self.target_class or 'None'}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
        cv2.putText(dash, f"STATUS: {status}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        if distance:
            cv2.putText(dash, f"DIST: {distance:.2f}m", (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        return np.hstack((rgb, dash))

    def is_obstacle_ahead(self, scan):
        if not scan:
            return False
        ranges = np.array(scan.ranges)
        front = ranges[len(ranges)//2 - 40 : len(ranges)//2 + 40]
        min_dist = np.nanmin(front)
        return min_dist < 0.45

    def estimate_distance(self, depth, cx, cy, size=13):
        half = size // 2
        region = depth[max(0, cy-half):cy+half+1, max(0, cx-half):cx+half+1]
        valid = region[(region > 0.2) & np.isfinite(region)]
        return float(np.median(valid)) if len(valid) > 5 else None

    def display_loop(self):
        cv2.namedWindow("Object Hunter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Hunter", 1600, 900)

        while rclpy.ok() and self.running:
            with self.frame_lock:
                rgb = self.latest_rgb.copy() if self.latest_rgb is not None else None
                depth = self.latest_depth
                scan = self.latest_scan

            if rgb is not None:
                result = self.process_frame(rgb, depth, scan)
                cv2.imshow("Object Hunter", result)
            else:
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...", (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
                cv2.imshow("Object Hunter", placeholder)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectHunterNode()
    try:
        node.display_loop()
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


