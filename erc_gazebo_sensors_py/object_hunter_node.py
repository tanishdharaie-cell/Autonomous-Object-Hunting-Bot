import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time

from ultralytics import YOLO


class ObjectHunterNode(Node):
    def __init__(self):
        super().__init__('object_hunter')
        self.model = YOLO("yolov8m.pt")
        self.get_logger().info("YOLOv8m loaded")

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
        self.stop_distance = 0.65

        # Tracking
        self.tracker = None
        self.tracked_bbox = None
        self.lost_frames = 0

        self.running = True

        self.spin_thread = threading.Thread(target=self.spin_thread_func, daemon=True)
        self.spin_thread.start()

        self.input_thread = threading.Thread(target=self.input_thread_func, daemon=True)
        self.input_thread.start()

        self.get_logger().info("Object Hunter Ready!")

    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def input_thread_func(self):
        while rclpy.ok() and self.running:
            if not self.mission_active:
                print("\n" + "="*60)
                target = input("Enter target (chair / person / cone / fridge): ").strip().lower()
                if target:
                    self.target_class = target
                    self.reset_tracking()
                    self.mission_active = True
                    print(f"🎯 Searching for: {target}")
            time.sleep(0.4)

    def reset_tracking(self):
        self.tracker = None
        self.tracked_bbox = None
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
        if not self.target_class:
            return False
        mapping = {
            "chair": ["chair"],
            "person": ["person"],
            "fridge": ["refrigerator", "fridge"],
            "cone": ["cone", "traffic light"],
        }
        allowed = mapping.get(self.target_class, [self.target_class])
        return any(a in class_name for a in allowed)

    def process_frame(self, rgb, depth, scan):
        h, w = rgb.shape[:2]
        results = self.model(rgb, conf=0.30, imgsz=640, verbose=False)

        target_detected = False
        best_bbox = None
        best_conf = 0.0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.model.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])

                if self.is_target_match(class_name) and conf > best_conf:
                    target_detected = True
                    best_conf = conf
                    best_bbox = (x1, y1, x2, y2)

                # Draw all detections
                color = (0, 255, 0) if self.is_target_match(class_name) else (0, 165, 255)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(rgb, f"{class_name} {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Initialize tracker on first good detection
        if target_detected and best_bbox and self.tracker is None:
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(rgb, best_bbox)
            self.tracked_bbox = best_bbox
            print(f"✅ Locked onto {self.target_class}! Starting approach...")

        # Update tracker
        tracked_ok = False
        cx = cy = None
        if self.tracker is not None:
            ok, box = self.tracker.update(rgb)
            if ok:
                tracked_ok = True
                self.lost_frames = 0
                x, y, ww, hh = map(int, box)
                self.tracked_bbox = (x, y, x+ww, y+hh)
                cx = x + ww//2
                cy = y + hh//2
            else:
                self.lost_frames += 1

        # Re-seed tracker if YOLO sees the target again
        if not tracked_ok and target_detected and best_bbox:
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(rgb, best_bbox)
            self.tracked_bbox = best_bbox
            tracked_ok = True
            cx = (best_bbox[0] + best_bbox[2]) // 2
            cy = (best_bbox[1] + best_bbox[3]) // 2

        if tracked_ok and self.tracked_bbox:
            x1, y1, x2, y2 = self.tracked_bbox
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)  # Blue = tracked target

        # Control
        twist = Twist()
        status = "SEARCHING"
        distance = None

        if tracked_ok and cx is not None:
            distance = self.estimate_distance(depth, cx, cy)

        if self.mission_active and self.target_class:
            if tracked_ok and cx is not None:
                status = "TRACKING"
                error = (w // 2 - cx) / (w / 2.0)

                if distance and distance <= self.stop_distance:
                    status = "COMPLETED"
                    self.mission_active = False
                    self.reset_tracking()
                    twist.linear.x = twist.angular.z = 0.0
                    print(f"\n🎉 MISSION COMPLETED! Reached {self.target_class}")
                elif abs(error) > 0.15:
                    twist.angular.z = 0.9 * error   # Strong centering
                    twist.linear.x = 0.0
                else:
                    twist.linear.x = max(0.12, min(0.38, (distance or 2.0) * 0.18))
                    twist.angular.z = 0.4 * error   # Small correction while moving
            else:
                status = "SEARCHING"
                twist.angular.z = 0.5   # Rotate to search

        self.cmd_vel_pub.publish(twist)

        # Dashboard
        dash = np.zeros((h, 450, 3), dtype=np.uint8)
        cv2.putText(dash, "OBJECT HUNTER", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.putText(dash, f"TARGET: {self.target_class or 'None'}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
        cv2.putText(dash, f"STATUS: {status}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        if distance:
            cv2.putText(dash, f"DIST: {distance:.2f}m", (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        return np.hstack((rgb, dash))

    def estimate_distance(self, depth, cx, cy, size=11):
        if depth is None:
            return None
        half = size // 2
        h, w = depth.shape
        x1 = max(0, cx - half)
        x2 = min(w, cx + half + 1)
        y1 = max(0, cy - half)
        y2 = min(h, cy + half + 1)
        region = depth[y1:y2, x1:x2]
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
                cv2.putText(placeholder, "Waiting for camera feed...", (200, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
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
