import math
import numpy as np
import cv2
import apriltag
import time

class AprilTagDetector(object):
    def __init__(self, tag_dict, proximity_thresholds):
        self.tag_dict = tag_dict
        self.detector = apriltag.Detector()
        self.proximity_thresholds = proximity_thresholds

    def detect_apriltags(self, img_arr):
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray_img)
        return detections

    def is_tag_close(self, tag, img_shape):
        tag_id = tag.tag_id
        if tag_id in self.proximity_thresholds:
            tag_width = tag.corners[2][0] - tag.corners[0][0]
            img_height, img_width = img_shape[:2]
            proximity_threshold = self.proximity_thresholds[tag_id]
            if (tag_width / img_width > proximity_threshold):
                return True
        return False

    def draw_bounding_box(self, tag, img_arr):
        img_arr = np.ascontiguousarray(img_arr)
        for corner in tag.corners:
            cv2.circle(img_arr, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
        cv2.polylines(img_arr, [tag.corners.astype(int)], True, (0, 255, 0), 2)

class ZebraCrosswalkDetector(object):
    def __init__(self, detection_hz):
        self.detection_hz = detection_hz
        self.last_detection_time = 0

    def detect_crosswalk(self, img_arr, debug_visuals):
        current_time = time.time()
        if current_time - self.last_detection_time >= 1.0 / self.detection_hz:
            self.last_detection_time = current_time
            gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_img, 175, 225, apertureSize=3)
            
            # Use Hough Line Transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=21, maxLineGap=5)
            if debug_visuals:
                cv2.imshow('Crosswalk', edges)
                cv2.waitKey(1)
            if lines is not None:
                # Filter lines to detect zebra crosswalk pattern
                vertical_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 10]
                return vertical_lines
        return []

    def draw_crosswalk_lines(self, lines, img_arr):
        img_arr = np.ascontiguousarray(img_arr)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)

class TurnManager(object):
    def __init__(self, turn_duration, initial_wait_time):
        self.turn_duration = turn_duration
        self.turn_start_time = 0
        self.initial_wait_time = initial_wait_time  # Wait time before executing the turn
        self.wait_start_time = 0

    def start_turn(self):
        self.turn_start_time = time.time()
        self.wait_start_time = time.time()

    def is_waiting(self):
        return time.time() - self.wait_start_time < self.initial_wait_time

    def is_turning(self):
        return time.time() - self.turn_start_time < self.turn_duration

class ProceedManager(object):
    def __init__(self, correction_duration, straight_duration):
        self.correction_duration = correction_duration
        self.straight_duration = straight_duration
        self.proceed_start_time = 0

    def start_proceed(self):
        self.proceed_start_time = time.time()

    def is_correcting(self):
        return time.time() - self.proceed_start_time < self.correction_duration

    def is_going_straight(self):
        elapsed_time = time.time() - self.proceed_start_time
        return self.correction_duration <= elapsed_time < (self.correction_duration + self.straight_duration)

class FIRAEngine(object):
    def __init__(self, tag_dict, proximity_thresholds, apriltag_hz, zebra_hz, top_crop_ratio, stop_duration=5, turn_duration=2, wait_duration=3.0, turn_initial_wait_duration=1.0, proceed_correction_duration=1.0, proceed_straight_duration=1.0, debug_visuals=True, debug=False):
        print("Initializing FIRA engine...")
        self.apriltag_detector = AprilTagDetector(tag_dict, proximity_thresholds)
        self.zebra_crosswalk_detector = ZebraCrosswalkDetector(zebra_hz)
        self.turn_manager = TurnManager(turn_duration,turn_initial_wait_duration)
        self.proceed_manager = ProceedManager(proceed_correction_duration, proceed_straight_duration)
        self.debug_visuals = debug_visuals
        self.debug = debug
        self.wait_duration = wait_duration

        self.state = 'idle'
        self.stop_start_time = 0
        self.stop_duration = stop_duration
        self.last_apriltag_detection_time = 0
        self.apriltag_hz = apriltag_hz
        self.top_crop_ratio = top_crop_ratio

        # Variable to track the detected AprilTag type
        self.detected_apriltag = None

        if self.debug:
            print("FIRA engine running...")

    # scalate image to 640x480
    def scale_image(self, img, width=640, height=480):
        img_height, img_width, _ = img.shape
        if img_height != height or img_width != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img

    def crop_image(self, img, crop_ratio=None):
        if crop_ratio is None:
            crop_ratio = self.top_crop_ratio
        img_height, img_width, _ = img.shape
        cropped_img = img[int(img_height * crop_ratio):, :]
        return cropped_img
    
    def detect_dashed_center_line(self, img):
        img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 150, 250, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=30)

        if lines is not None:
            img_height, img_width = img.shape[:2]
            center_x = img_width // 2
            best_line = None
            min_distance = float('inf')
            correction_angle = 0  # Default angle is 0 (no correction needed)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_center_x = (x1 + x2) // 2
                distance_to_center = abs(line_center_x - center_x)

                # Filtrar líneas casi verticales y cercanas al centro
                if abs(x2 - x1) < 30 and distance_to_center < min_distance:
                    best_line = (x1, y1, x2, y2)
                    min_distance = distance_to_center

                    # Calcular el ángulo de inclinación
                    delta_y = y2 - y1
                    delta_x = x2 - x1
                    correction_angle = math.degrees(math.atan2(delta_y, delta_x))  # Convert to degrees

            if best_line:
                x1, y1, x2, y2 = best_line
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Dibujar en verde
                if self.debug_visuals:
                    # Mostrar el ángulo de corrección en la imagen
                    text_position = (50, 50)  # Coordenadas donde se mostrará el texto
                    cv2.putText(img, f"Angle: {correction_angle:.2f} deg", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return correction_angle, img

    def detect_lane_and_correction(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 200, 300, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

        if self.debug_visuals:
            img = edges.copy()
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if self.debug_visuals:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            right_lane = max(lines, key=lambda line: line[0][0])
            x1, y1, x2, y2 = right_lane[0]

            correction_angle = (x1 + x2) / 2 - img.shape[1] / 2

            if self.debug_visuals:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.arrowedLine(img, (img.shape[1] // 2, img.shape[0]), (int((x1 + x2) / 2), int((y1 + y2) / 2)), (255, 0, 0), 2)

            return correction_angle / img.shape[1], img
        return 0, img

    def detect_apriltags_and_update_state(self, img_arr, current_time, throttle, angle):
        apriltag_detections = self.apriltag_detector.detect_apriltags(img_arr)
        for tag in apriltag_detections:
            if self.apriltag_detector.is_tag_close(tag, img_arr.shape):
                tag_name = self.apriltag_detector.tag_dict.get(tag.tag_id, 'UNKNOWN')
                if self.debug:
                    print("Detected tag: " + str(tag.tag_id))
                self.detected_apriltag = tag_name
                if tag_name in ['STOP', 'DEAD_END']:
                    self.state = 'stop'
                    self.stop_start_time = current_time
                    if self.debug_visuals:
                        self.apriltag_detector.draw_bounding_box(tag, img_arr)
                    if self.debug:
                        print(f"AprilTag detected: {tag_name} - Stop")
                    break
                elif tag_name in ['TURN_LEFT', 'TURN_RIGHT', 'FORWARD']:
                    self.state = 'wait-for-crosswalk'
                    self.saved_angle = angle
                    self.saved_throttle = throttle
                    if self.debug_visuals:
                        self.apriltag_detector.draw_bounding_box(tag, img_arr)
                    if self.debug:
                        print(f"AprilTag detected: {tag_name} - {self.detected_apriltag}")
                    break
        return angle, throttle, img_arr

    def run(self, angle, throttle, input_img_arr):
        try:
            current_time = time.time()
            show_img = self.scale_image(input_img_arr)
            if show_img is None:
                return angle, throttle, input_img_arr

            if self.state == 'stop':
                if current_time - self.stop_start_time >= self.stop_duration:
                    self.state = 'idle'
                return 0, 0, input_img_arr

            if self.state == 'wait-for-crosswalk':
                crosswalk_lines = self.zebra_crosswalk_detector.detect_crosswalk(show_img, self.debug_visuals)
                if self.debug_visuals:
                    for line in crosswalk_lines:
                        for x1, y1, x2, y2 in line:
                            cv2.line(show_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if len(crosswalk_lines) >= 5:
                    self.state = 'wait-at-crosswalk'
                    self.stop_start_time = current_time
                    if self.debug_visuals:
                        self.zebra_crosswalk_detector.draw_crosswalk_lines(crosswalk_lines, show_img)
                    if self.debug:
                        print("Zebra crosswalk detected")
                    return 0, 0, input_img_arr
                return angle, throttle, input_img_arr

            if self.state == 'wait-at-crosswalk':
                if current_time - self.stop_start_time >= self.wait_duration:
                    if self.detected_apriltag == 'TURN_LEFT':
                        self.turn_manager.start_turn()
                        self.state = 'turn_left'
                    elif self.detected_apriltag == 'TURN_RIGHT':
                        self.turn_manager.start_turn()                    
                        self.state = 'turn_right'
                    elif self.detected_apriltag == 'FORWARD':
                        self.proceed_manager.start_proceed()
                        self.state = 'proceeding'
                    self.detected_apriltag = None
                    return 0, 0, input_img_arr
                return 0, 0, input_img_arr

            if self.state in ['turn_left', 'turn_right']:
                if self.turn_manager.is_waiting():
                    return 0, 1, input_img_arr
                if self.turn_manager.is_turning():
                    turn_angle = -1 if self.state == 'turn_left' else 1
                    turn_throttle = 1
                    return turn_angle, turn_throttle, input_img_arr
                else:
                    self.state = 'idle'
                    self.detected_apriltag = None
                    return angle, throttle, input_img_arr

            if self.state == 'proceeding':
                if self.debug_visuals:
                    cv2.imshow('Street', show_img)
                    cv2.waitKey(1)
                if self.proceed_manager.is_correcting():
                    # test
                    correction_angle, show_img = self.detect_dashed_center_line(show_img)
                    # correction_angle, show_img = self.detect_lane_and_correction(show_img)
                    if self.debug_visuals:
                        cv2.imshow('Street', show_img)
                        cv2.waitKey(1)
                    return correction_angle, 1, input_img_arr
                elif self.proceed_manager.is_going_straight():
                    return 0, 1, input_img_arr
                else:
                    self.state = 'idle'
                    self.detected_apriltag = None
                    return 0, 1, input_img_arr

            if self.state == 'idle':

                # Determine if it's time to detect AprilTags
                if current_time - self.last_apriltag_detection_time >= 1.0 / self.apriltag_hz:
                    self.last_apriltag_detection_time = current_time
                    if self.debug:
                        print("Searching for AprilTag...")
                    # Detect AprilTags
                    angle, throttle, show_img = self.detect_apriltags_and_update_state(show_img, current_time, throttle, angle)
                    if angle and throttle:
                        return angle, throttle, input_img_arr

            # Show current detection results if debug_visuals is enabled
            if self.debug_visuals:
                cv2.imshow('Realsense', show_img)
                cv2.imshow('Street', show_img)
                cv2.waitKey(1)
                
            return angle, throttle, input_img_arr
        except Exception as e:
            print(f"Error in FIRAEngine run: {e}")