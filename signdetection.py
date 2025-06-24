import numpy as np
import cv2
import time
import collections
import apriltag
import os
from PIL import Image
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from donkeycar.parts.velocity import VelocityUnnormalize

# Utils
def is_display_available():
    return os.environ.get("DISPLAY") is not None

def safe_imshow(winname, img):
    try:
        cv2.imshow(winname, img)
        cv2.waitKey(1)
    except cv2.error:
        pass

class UrbanTrackController:
    def __init__(self, tag_dict, proximity_thresholds, apriltag_hz=2, stop_duration=5, turn_duration=2,
                 correction_duration=1.5, forward_duration=1.5, speed_scale=1.0, debug_visuals=True, debug=False):

        self.debug = debug
        self.debug_visuals = debug_visuals if is_display_available() else False

        # AprilTag
        self.tag_dict = tag_dict
        self.proximity_thresholds = proximity_thresholds
        self.detector = apriltag.Detector()
        self.last_apriltag_time = 0
        self.apriltag_hz = apriltag_hz

        # Stop Sign Detection
        self.STOP_SIGN_CLASS_ID = 12
        self.min_score = 0.5
        self.engine = DetectionEngine("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
        self.labels = dataset_utils.read_label_file("coco_labels.txt")

        # States
        self.state = 'idle'
        self.stop_start_time = 0
        self.turn_start_time = 0
        self.correction_start_time = 0
        self.detected_apriltag = None

        # Durations
        self.stop_duration = stop_duration
        self.turn_duration = turn_duration
        self.correction_duration = correction_duration
        self.forward_duration = forward_duration

        # Speed control
        self.speed_scale = speed_scale
        self.velocity_map = VelocityUnnormalize(min_speed=0.2, max_speed=1.2)
        self.turn_speed = 0.3
        self.forward_speed = 0.5
        self.correction_speed = 0.4

    def detect_stop_sign(self, img_arr):
        img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        results = self.engine.detect_with_image(img, threshold=self.min_score, keep_aspect_ratio=True,
                                                relative_coord=False, top_k=3)
        for obj in results:
            if obj.label_id == self.STOP_SIGN_CLASS_ID:
                if self.debug:
                    print(f"Stop sign detected with score {obj.score}")
                return obj
        return None

    def detect_apriltags(self, img_arr):
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray_img)

    def is_tag_close(self, tag):
        tag_id = tag.tag_id
        if tag_id in self.proximity_thresholds:
            tag_width = tag.corners[2][0] - tag.corners[0][0]
            _, img_width = tag.corners.shape[:2]
            threshold = self.proximity_thresholds[tag_id]
            return tag_width / img_width > threshold
        return False

    def detect_lane_and_correction(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 150, 250)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=20)

        if self.debug_visuals:
            safe_imshow("Lane", edges)

        if lines is not None:
            img_center_x = img.shape[1] / 2
            best_line = min(lines, key=lambda l: abs(((l[0][0] + l[0][2]) / 2) - img_center_x))
            x1, y1, x2, y2 = best_line[0]
            correction = ((x1 + x2) / 2 - img_center_x) / img_center_x
            return correction
        return 0

    def run(self, img_arr, throttle):
        current_time = time.time()
        angle = 0  # default

        # Check STOP sign
        stop_obj = self.detect_stop_sign(img_arr)
        if stop_obj and self.state == 'idle':
            self.state = 'stop'
            self.stop_start_time = current_time
            return 0, 0, img_arr

        # FSM states
        if self.state == 'stop':
            if current_time - self.stop_start_time >= self.stop_duration:
                self.state = 'idle'
            return 0, 0, img_arr

        elif self.state == 'turning':
            if current_time - self.turn_start_time < self.turn_duration:
                angle = -1 if self.detected_apriltag == 'TURN_LEFT' else 1
                throttle = self.velocity_map.run(self.turn_speed) * self.speed_scale
                return angle, throttle, img_arr
            else:
                self.state = 'correction'
                self.correction_start_time = current_time
                return 0, 0, img_arr

        elif self.state == 'correction':
            if current_time - self.correction_start_time < self.correction_duration:
                angle = self.detect_lane_and_correction(img_arr)
                throttle = self.velocity_map.run(self.correction_speed) * self.speed_scale
                return angle, throttle, img_arr
            else:
                self.state = 'idle'
                return 0, throttle, img_arr

        elif self.state == 'forward':
            if current_time - self.correction_start_time < self.forward_duration:
                angle = self.detect_lane_and_correction(img_arr)
                throttle = self.velocity_map.run(self.forward_speed) * self.speed_scale
                return angle, throttle, img_arr
            else:
                self.state = 'idle'
                return 0, throttle, img_arr

        elif self.state == 'idle':
            # Detect AprilTags periodically
            if current_time - self.last_apriltag_time > 1.0 / self.apriltag_hz:
                self.last_apriltag_time = current_time
                detections = self.detect_apriltags(img_arr)
                for tag in detections:
                    if self.is_tag_close(tag):
                        tag_name = self.tag_dict.get(tag.tag_id, 'UNKNOWN')
                        self.detected_apriltag = tag_name
                        if self.debug:
                            print(f"AprilTag detected: {tag.tag_id} -> {tag_name}")
                        if tag_name == 'STOP' or tag_name == 'DEAD_END':
                            self.state = 'stop'
                            self.stop_start_time = current_time
                        elif tag_name in ['TURN_LEFT', 'TURN_RIGHT']:
                            self.state = 'turning'
                            self.turn_start_time = current_time
                        elif tag_name == 'FORWARD':
                            self.state = 'forward'
                            self.correction_start_time = current_time
                        return 0, 0, img_arr

        return angle, throttle, img_arr
