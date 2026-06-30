import math
import threading
import time
from enum import Enum

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from ultralytics import YOLO


class MissionState(Enum):
    IDLE = 0
    SEARCH = 1
    TRACK = 2
    APPROACH = 3
    RECOVER = 4
    COMPLETE = 5


class ObjectHunterNode(Node):
    def __init__(self):
        super().__init__('object_hunter')

        model_path = 'yolov8m.pt'
        self.model = YOLO(model_path)
        self.get_logger().info(f'Loaded detector: {model_path}')

        self.bridge = CvBridge()
        self.create_subscription(Image, 'camera/image', self.image_callback, 10)
        self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.frame_lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_scan = None
        self.latest_detections = []

        self.target_class = None
        self.target_track_id = None
        self.state = MissionState.IDLE
        self.running = True

        self.stop_distance = 0.80
        self.goal_tolerance = 0.10
        self.front_stop_distance = 0.55
        self.front_slow_distance = 0.90
        self.target_reacquire_timeout = 1.2
        self.target_drop_timeout = 3.0
        self.search_forward_clearance = 1.2
        self.horizontal_fov_deg = 69.0

        self.max_linear_speed = 0.25
        self.max_angular_speed = 0.75
        self.max_linear_accel = 0.20
        self.max_angular_accel = 1.0
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.last_cmd_time = time.time()

        self.search_direction = 1.0
        self.search_phase_started = time.time()
        self.last_target_seen_time = 0.0

        self.class_aliases = {
            'chair': {'chair'},
            'person': {'person'},
            'fridge': {'refrigerator'},
            'cone': {'traffic cone', 'cone'},
        }

        self.class_thresholds = {
            'person': 0.55,
            'chair': 0.45,
            'refrigerator': 0.45,
        }

        self.kf = self._build_target_kalman()
        self.kf_initialized = False
        self.smoothed_target = None
        self.last_status_text = 'IDLE'

        self.spin_thread = threading.Thread(target=self.spin_thread_func, daemon=True)
        self.detect_thread = threading.Thread(target=self.detect_loop, daemon=True)
        self.input_thread = threading.Thread(target=self.input_thread_func, daemon=True)

        self.spin_thread.start()
        self.detect_thread.start()
        self.input_thread.start()

        self.get_logger().info('Object Hunter ready')

    def _build_target_kalman(self):
        kf = cv2.KalmanFilter(6, 3)
        # state = [cx, cy, dist, vx, vy, vdist]
        kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.diag([1.0, 1.0, 0.12, 4.0, 4.0, 0.5]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([20.0, 20.0, 0.08]).astype(np.float32)
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 5.0
        return kf

    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def input_thread_func(self):
        while rclpy.ok() and self.running:
            if self.state == MissionState.IDLE:
                print('\n' + '=' * 60)
                target = input('Enter target (chair / person / fridge / cone): ').strip().lower()
                if target:
                    self.target_class = target
                    if not self.target_supported_by_model():
                        supported = ', '.join(sorted(set(name.lower() for name in self.model.names.values())))
                        print(
                            f"❌ '{target}' is not a native class in the current YOLO model.\n"
                            f"Use a custom-trained model for that object.\n"
                            f"Supported classes include: {supported}"
                        )
                        self.target_class = None
                        time.sleep(0.2)
                        continue

                    self.reset_target_state()
                    self.state = MissionState.SEARCH
                    print(f'🎯 Searching for: {target}')
            time.sleep(0.2)

    def reset_target_state(self):
        self.target_track_id = None
        self.kf = self._build_target_kalman()
        self.kf_initialized = False
        self.smoothed_target = None
        self.last_target_seen_time = 0.0

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.frame_lock:
                self.latest_rgb = frame
        except Exception as exc:
            self.get_logger().warning(f'RGB conversion failed: {exc}')

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            with self.frame_lock:
                self.latest_depth = depth
        except Exception as exc:
            self.get_logger().warning(f'Depth conversion failed: {exc}')

    def scan_callback(self, msg):
        with self.frame_lock:
            self.latest_scan = msg

    def canonical_classes(self):
        if not self.target_class:
            return set()
        return self.class_aliases.get(self.target_class, {self.target_class})

    def target_supported_by_model(self):
        model_classes = {name.lower() for name in self.model.names.values()}
        return any(name in model_classes for name in self.canonical_classes())

    def is_target_match(self, class_name):
        return class_name in self.canonical_classes()

    def detect_loop(self):
        while rclpy.ok() and self.running:
            if self.state == MissionState.IDLE:
                time.sleep(0.1)
                continue

            with self.frame_lock:
                frame = None if self.latest_rgb is None else self.latest_rgb.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()
                scan = self.latest_scan

            if frame is None:
                time.sleep(0.03)
                continue

            detections = self.run_detection_and_tracking(frame, depth, scan)
            with self.frame_lock:
                self.latest_detections = detections

            time.sleep(0.06)  # ~16 Hz detect/track loop

    def run_detection_and_tracking(self, rgb, depth, scan):
        target_classes = sorted(self.canonical_classes())
        class_ids = [idx for idx, name in self.model.names.items() if name.lower() in target_classes] if target_classes else None

        results = self.model.track(
            source=rgb,
            persist=True,
            tracker='bytetrack.yaml',
            conf=0.30,
            iou=0.45,
            imgsz=640,
            classes=class_ids if class_ids else None,
            verbose=False,
        )

        detections = []
        result = results[0]
        boxes = result.boxes
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = self.model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            threshold = self.class_thresholds.get(class_name, 0.45)
            if conf < threshold:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_m = self.estimate_distance(depth, (x1, y1, x2, y2))
            bearing = self.pixel_to_bearing(cx, rgb.shape[1])
            scan_m = self.range_at_bearing(scan, bearing)
            fused_distance = self.fuse_range(depth_m, scan_m)

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'cx': cx,
                'cy': cy,
                'conf': conf,
                'class_name': class_name,
                'track_id': track_id,
                'distance': fused_distance,
                'depth_distance': depth_m,
                'scan_distance': scan_m,
                'bearing': bearing,
            })

        return detections

    def pixel_to_bearing(self, cx, width):
        norm = (cx - (width / 2.0)) / (width / 2.0)
        return math.radians((self.horizontal_fov_deg / 2.0) * norm)

    def range_at_bearing(self, scan, bearing, window_deg=4.0):
        if scan is None or not scan.ranges:
            return None
        if scan.angle_increment == 0.0:
            return None

        half_window = math.radians(window_deg)
        a0 = bearing - half_window
        a1 = bearing + half_window
        i0 = max(0, int((a0 - scan.angle_min) / scan.angle_increment))
        i1 = min(len(scan.ranges) - 1, int((a1 - scan.angle_min) / scan.angle_increment))
        if i1 < i0:
            i0, i1 = i1, i0

        vals = []
        for i in range(i0, i1 + 1):
            r = scan.ranges[i]
            if scan.range_min < r < scan.range_max and math.isfinite(r):
                vals.append(r)

        return float(np.median(vals)) if vals else None

    def front_clearance(self, scan, window_deg=12.0):
        return self.range_at_bearing(scan, 0.0, window_deg)

    def estimate_distance(self, depth, bbox):
        if depth is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = depth.shape[:2]

        bw = max(6, x2 - x1)
        bh = max(6, y2 - y1)

        # central-lower ROI is usually more stable than exact center pixel
        rx1 = max(0, x1 + int(0.25 * bw))
        rx2 = min(w, x2 - int(0.25 * bw))
        ry1 = max(0, y1 + int(0.45 * bh))
        ry2 = min(h, y2 - int(0.10 * bh))

        region = depth[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return None

        valid = region[np.isfinite(region) & (region > 0.15) & (region < 8.0)]
        if valid.size < 15:
            return None

        return float(np.median(valid))

    def fuse_range(self, depth_m, scan_m):
        vals = [v for v in [depth_m, scan_m] if v is not None and math.isfinite(v)]
        return float(min(vals)) if vals else None

    def choose_target(self, detections, width, height):
        if not detections:
            return None

        target_dets = [d for d in detections if self.is_target_match(d['class_name'])]
        if not target_dets:
            return None

        # keep same ID if possible
        if self.target_track_id is not None:
            for det in target_dets:
                if det['track_id'] == self.target_track_id:
                    return det

        pred = self.predict_target()
        pred_cx = width / 2.0 if pred is None else float(pred[0])
        pred_cy = height / 2.0 if pred is None else float(pred[1])
        pred_dist = None if pred is None else float(pred[2])

        def det_score(det):
            area = (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
            center_penalty = abs(det['cx'] - pred_cx) / max(1.0, width)
            vertical_penalty = abs(det['cy'] - pred_cy) / max(1.0, height)
            dist_penalty = 0.0
            if pred_dist is not None and det['distance'] is not None:
                dist_penalty = min(1.0, abs(det['distance'] - pred_dist) / 2.0)
            return (2.0 * det['conf']) + (0.000002 * area) - center_penalty - 0.5 * vertical_penalty - 0.6 * dist_penalty

        target_dets.sort(key=det_score, reverse=True)
        return target_dets[0]

    def predict_target(self):
        if not self.kf_initialized:
            return None
        pred = self.kf.predict()
        self.smoothed_target = pred.copy()
        return pred.reshape(-1)

    def correct_target(self, cx, cy, dist):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(dist)]])
        if not self.kf_initialized:
            self.kf.statePost = np.array([[cx], [cy], [dist], [0.0], [0.0], [0.0]], dtype=np.float32)
            self.kf_initialized = True
            self.smoothed_target = self.kf.statePost.copy()
        else:
            self.smoothed_target = self.kf.correct(measurement)
        return self.smoothed_target.reshape(-1)

    def is_goal_reached(self, distance, front_clear):
        if distance is not None and distance <= (self.stop_distance + self.goal_tolerance):
            return True
        if front_clear is not None and front_clear <= (self.front_stop_distance + 0.03):
            return True
        return False

    def search_behavior(self, twist, front_clear, now):
        elapsed = now - self.search_phase_started
        if elapsed > 4.0:
            self.search_direction *= -1.0
            self.search_phase_started = now

        if front_clear is not None and front_clear < self.front_stop_distance:
            twist.linear.x = 0.0
            twist.angular.z = -0.5 * self.search_direction
        elif front_clear is not None and front_clear > self.search_forward_clearance and elapsed > 2.0:
            twist.linear.x = 0.08
            twist.angular.z = 0.25 * self.search_direction
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.45 * self.search_direction

    def update_state_machine(self, rgb, depth, scan, detections):
        h, w = rgb.shape[:2]
        target = self.choose_target(detections, w, h)
        now = time.time()

        if target is not None:
            self.target_track_id = target['track_id']
            self.last_target_seen_time = now
            measured_dist = target['distance'] if target['distance'] is not None else 3.0
            filt = self.correct_target(target['cx'], target['cy'], measured_dist)
            filt_cx, filt_cy, filt_dist = float(filt[0]), float(filt[1]), float(filt[2])
        else:
            filt = self.predict_target()
            if filt is not None:
                filt_cx, filt_cy, filt_dist = float(filt[0]), float(filt[1]), float(filt[2])
            else:
                filt_cx, filt_cy, filt_dist = w / 2.0, h / 2.0, None

        front_clear = self.front_clearance(scan)
        twist = Twist()

        if self.state == MissionState.SEARCH:
            if target is not None:
                self.state = MissionState.TRACK
            else:
                self.search_behavior(twist, front_clear, now)

        if self.state == MissionState.TRACK:
            if target is None:
                self.state = MissionState.RECOVER
            else:
                error = ((w / 2.0) - filt_cx) / (w / 2.0)
                if abs(error) > 0.08:
                    twist.angular.z = np.clip(0.9 * error, -self.max_angular_speed, self.max_angular_speed)
                    twist.linear.x = 0.0
                else:
                    self.state = MissionState.APPROACH

        if self.state == MissionState.APPROACH:
            if target is None:
                self.state = MissionState.RECOVER
            else:
                error = ((w / 2.0) - filt_cx) / (w / 2.0)
                distance = target['distance'] if target['distance'] is not None else filt_dist

                if self.is_goal_reached(distance, front_clear):
                    self.state = MissionState.COMPLETE
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    print(f'\n🎉 OBJECT FOUND: reached {self.target_class}')
                elif front_clear is not None and front_clear <= self.front_stop_distance:
                    self.state = MissionState.RECOVER
                else:
                    distance = 2.5 if distance is None else distance
                    desired = np.clip(0.18 * (distance - self.stop_distance), 0.0, self.max_linear_speed)

                    if front_clear is not None and front_clear < self.front_slow_distance:
                        desired *= max(
                            0.15,
                            (front_clear - self.front_stop_distance) / max(0.05, self.front_slow_distance - self.front_stop_distance)
                        )

                    if abs(error) > 0.25:
                        desired *= 0.2
                    elif abs(error) > 0.12:
                        desired *= 0.5

                    twist.linear.x = desired
                    twist.angular.z = np.clip(0.65 * error, -self.max_angular_speed, self.max_angular_speed)

        if self.state == MissionState.RECOVER:
            time_since_seen = now - self.last_target_seen_time if self.last_target_seen_time else 999.0
            if target is not None:
                self.state = MissionState.TRACK
            elif time_since_seen < self.target_reacquire_timeout and self.kf_initialized:
                error = ((w / 2.0) - filt_cx) / (w / 2.0)
                twist.angular.z = np.clip(0.6 * error, -0.5, 0.5)
                twist.linear.x = 0.0
            elif time_since_seen < self.target_drop_timeout:
                twist.angular.z = 0.35 * self.search_direction
                twist.linear.x = 0.0
            else:
                self.target_track_id = None
                self.state = MissionState.SEARCH
                self.search_phase_started = now
                self.search_behavior(twist, front_clear, now)

        if self.state == MissionState.COMPLETE:
            self.last_status_text = 'COMPLETE'
            self.state = MissionState.IDLE
            self.target_track_id = None
            self.target_class = None
            self.kf_initialized = False
            return self.rate_limit_twist(Twist())

        if self.state == MissionState.IDLE:
            self.last_status_text = 'IDLE'
            return self.rate_limit_twist(Twist())

        self.last_status_text = self.state.name
        return self.rate_limit_twist(twist)

    def rate_limit_twist(self, twist):
        now = time.time()
        dt = max(0.02, now - self.last_cmd_time)
        self.last_cmd_time = now

        max_lin_step = self.max_linear_accel * dt
        max_ang_step = self.max_angular_accel * dt

        lin = self._ramp(self.prev_linear, twist.linear.x, max_lin_step)
        ang = self._ramp(self.prev_angular, twist.angular.z, max_ang_step)

        out = Twist()
        out.linear.x = float(np.clip(lin, -self.max_linear_speed, self.max_linear_speed))
        out.angular.z = float(np.clip(ang, -self.max_angular_speed, self.max_angular_speed))

        self.prev_linear = out.linear.x
        self.prev_angular = out.angular.z
        return out

    @staticmethod
    def _ramp(current, desired, step):
        if desired > current + step:
            return current + step
        if desired < current - step:
            return current - step
        return desired

    def annotate(self, rgb, detections, commanded_twist):
        canvas = rgb.copy()
        selected = self.choose_target(detections, rgb.shape[1], rgb.shape[0])

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            is_selected = selected is not None and det['bbox'] == selected['bbox']
            color = (255, 0, 0) if is_selected else (0, 165, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} id={det['track_id']} {det['conf']:.2f}"
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if det['distance'] is not None:
                cv2.putText(canvas, f"{det['distance']:.2f}m", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if self.kf_initialized and self.smoothed_target is not None:
            cx = int(float(self.smoothed_target[0]))
            cy = int(float(self.smoothed_target[1]))
            cv2.circle(canvas, (cx, cy), 8, (0, 255, 255), -1)
            cv2.line(canvas, (canvas.shape[1] // 2, canvas.shape[0]), (cx, cy), (0, 255, 255), 2)

        dash = np.zeros((canvas.shape[0], 430, 3), dtype=np.uint8)
        lines = [
            'OBJECT HUNTER',
            f'Target: {self.target_class or "None"}',
            f'State: {self.last_status_text}',
            f'Locked ID: {self.target_track_id}',
            f'cmd.v: {commanded_twist.linear.x:.2f} m/s',
            f'cmd.w: {commanded_twist.angular.z:.2f} rad/s',
        ]
        y = 40
        for i, line in enumerate(lines):
            scale = 0.9 if i == 0 else 0.7
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(dash, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            y += 42

        return np.hstack((canvas, dash))

    def display_loop(self):
        cv2.namedWindow('Object Hunter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Hunter', 1600, 900)

        while rclpy.ok() and self.running:
            with self.frame_lock:
                rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()
                scan = self.latest_scan
                detections = list(self.latest_detections)

            if rgb is None:
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(placeholder, 'Waiting for camera feed...', (220, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow('Object Hunter', placeholder)
            else:
                cmd = self.update_state_machine(rgb, depth, scan, detections)
                self.cmd_vel_pub.publish(cmd)
                annotated = self.annotate(rgb, detections, cmd)
                cv2.imshow('Object Hunter', annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.cmd_vel_pub.publish(Twist())
        for th in [self.spin_thread, self.detect_thread, self.input_thread]:
            if th.is_alive():
                th.join(timeout=1.0)


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
