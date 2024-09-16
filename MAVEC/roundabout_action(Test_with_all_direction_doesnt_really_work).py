import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import json
import time
from collections import deque
import numpy as np
import cv2
from scipy.signal import find_peaks
from dataclasses import dataclass
from enum import Enum

class LaneType(Enum):
    LEFT_VERTICAL = 1
    RIGHT_VERTICAL = 2
    TOP_HORIZONTAL = 3
    
@dataclass
class LaneHypothesis:
    start_x: int
    start_y: int
    direction: str
    traced_points: list
    confidence: float
    lane_type: LaneType
    
class RoundaboutNavigationNode(Node):
    def __init__(self):
        super().__init__('roundabout_navigation_node')
        self.bridge = CvBridge()

        # Subscriptions
        self.create_subscription(Image, '/mavec/warped', self.warped_image_callback, 10)
        self.create_subscription(Float64MultiArray, '/action/roundabout', self.action_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.roundabout_pub = self.create_publisher(String, '/obs_det/roundabout', 10)
        self.image_pub = self.create_publisher(Image, '/lane_visualization', 10)

        # Initialize other attributes
        package_path = get_package_share_directory('mavec_action')
        self.save_folder = os.path.abspath(os.path.join(package_path, '../../../../roundabout_log/',f'roundabout_image_{int(time.time())}'))
        os.mkdir(self.save_folder)
        self.image_counter = 0
        
 
        # Initialize other attributes
        self.turn_frame = None
        self.is_turning = False
        self.turn_complete = False
        self.turning_movement = 0
        self.lane_previous = True
        self.first_frame_number = None
        self.top_horizontal_status = False
        self.left_vertical_status = False
        self.right_vertical_status = False
        
    def warped_image_callback(self, msg):
        current_frame = self.bridge.imgmsg_to_cv2(msg)
        print(f"top_horizontal_status{self.top_horizontal_status}")
        #print(f"turn_frame_status{self.turn_frame}")
        print(f"turn_complete_status{self.turn_complete}")
        if self.top_horizontal_status and self.turn_frame is None and not self.turn_complete:
            # Capture the turn frame when we first detect top_horizontal
            self.turn_frame = current_frame
            self.turning_movement = 0
            self.get_logger().info("Turn frame captured. Starting turn sequence.")
        
        if self.turn_frame is not None and not self.turn_complete:
            frame_to_process = self.turn_frame
        elif self.left_vertical_status or self.right_vertical_status:
            frame_to_process = current_frame
        else:
            frame_to_process = current_frame
            
        # Process the image
        hypotheses = self.detect_initial_lanes(frame_to_process)
        lanes = self.apply_sliding_window(frame_to_process, hypotheses)
        overlay_image, offset = self.finding_lane_center(frame_to_process, lanes)
        
        # Publish the visualization
        if overlay_image is not None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(overlay_image))
        
        # Save the image
        self.save_image(overlay_image)
        
        # Use the offset for navigation
        if offset is not None:
            if self.turn_frame is not None and not self.turn_complete:
                #self.create_timer(10.0,self.move_robot(offset, 1.0))
                self.create_timer(1.0, lambda: self.move_robot(offset,steering=1.0))
                self.turning_movement += 1
                print(f"turning_movement{self.turning_movement}")
                if self.turning_movement > 4:
                    self.turn_complete = True
                    #self.is_turning = False
                    self.turn_frame = None
                    self.get_logger().info("Turn complete. Switching to normal operation.")
                    print(f"turn_complete status: {self.turn_complete}")
                    print(f"turning_movement{self.turning_movement}")
            else:
                self.move_robot(offset, steering=2.0)        

    def threshold(self, image):
        return cv2.inRange(image, 200, 255)
    
    def detect_initial_lanes(self,image, margin=20, min_pixels=50, trace_length=40, max_hypotheses=5):
        if image is None:
            self.get_logger().error("Receivedn None image in detect_initial_lanes")
            return[]
        if len(image.shape) < 2:
            self.get_logger().error(f"Invalid image shape: {image.shape}")
            return []
        h, w = image.shape[:2]
        
        top = np.sum(image[:margin, :], axis=0)
        bottom = np.sum(image[-margin:, :], axis=0)
        left = np.sum(image[:, :margin], axis=1)
        right = np.sum(image[:, -margin:], axis=1)
        
        hypotheses = []
        
        def trace_edge(x, y, dx, dy):
            points = []
            for i in range(trace_length):
                nx, ny = int(x + i*dx), int(y + i*dy)
                if 0 <= nx < w and 0 <= ny < h and image[ny, nx] > 0:
                    points.append((nx, ny))
                else:
                    break
            return points
    
        def calculate_confidence(points, edge):
            if not points:
                return 0.0
            
            length_score = len(points) / trace_length
            avg_intensity = np.mean([image[y, x] for x, y in points]) / 255
            
            if edge in ['top', 'bottom']:
                continuity_score = 1 - (abs(points[-1][0] - points[0][0]) / w)
            elif edge in ['left', 'right']:
                continuity_score = 1 - (abs(points[-1][1] - points[0][1]) / h)
            else:
                continuity_score = 1.0
            
            if len(points) > 2:
                diffs = np.diff(np.array(points), axis=0)
                angle_changes = np.arctan2(diffs[:, 1], diffs[:, 0])
                curvature_consistency = 1 - (np.std(angle_changes) / np.pi)
            else:
                curvature_consistency = 1.0
            
            confidence = 0.3 * length_score + 0.3 * avg_intensity + 0.2 * continuity_score + 0.2 * curvature_consistency
            return confidence
        
        def add_hypothesis(edge, x, y):
            if edge == 'top':
                vertical_points = trace_edge(x, y, 0, 1)
                horizontal_points = trace_edge(x, y, 1, 0)
                
                vertical_confidence = calculate_confidence(vertical_points, edge)
                horizontal_confidence = calculate_confidence(horizontal_points, edge)
                
                if vertical_confidence > horizontal_confidence:
                    if x < w // 2:
                        hypotheses.append(LaneHypothesis(x, y, 'vertical', vertical_points, vertical_confidence, LaneType.LEFT_VERTICAL))
                    else:
                        hypotheses.append(LaneHypothesis(x, y, 'vertical', vertical_points, vertical_confidence, LaneType.RIGHT_VERTICAL))
                else:
                    hypotheses.append(LaneHypothesis(x, y, 'horizontal', horizontal_points, horizontal_confidence, LaneType.TOP_HORIZONTAL))
            
            elif edge == 'bottom':
                points = trace_edge(x, y, 0, -1)
                confidence = calculate_confidence(points, edge)
                if confidence > 0:
                    if x < w // 2:
                        hypotheses.append(LaneHypothesis(x, y, 'vertical', points, confidence, LaneType.LEFT_VERTICAL))
                    else:
                        hypotheses.append(LaneHypothesis(x, y, 'vertical', points, confidence, LaneType.RIGHT_VERTICAL))
            
            elif edge == 'left':
                vertical_points = trace_edge(x, y, 0, 1)
                horizontal_points = trace_edge(x, y, 1, 0)
                
                vertical_confidence = calculate_confidence(vertical_points, edge)
                horizontal_confidence = calculate_confidence(horizontal_points, edge)
                
                if vertical_confidence > horizontal_confidence:
                    hypotheses.append(LaneHypothesis(x, y, 'vertical', vertical_points, vertical_confidence, LaneType.LEFT_VERTICAL))
                else:
                    hypotheses.append(LaneHypothesis(x, y, 'horizontal', horizontal_points, horizontal_confidence, LaneType.TOP_HORIZONTAL))
            
            else:  # right edge
                vertical_points = trace_edge(x, y, 0, 1)
                horizontal_points = trace_edge(x, y, -1, 0)
                
                vertical_confidence = calculate_confidence(vertical_points, edge)
                horizontal_confidence = calculate_confidence(horizontal_points, edge)
                
                if vertical_confidence > horizontal_confidence:
                    hypotheses.append(LaneHypothesis(x, y, 'vertical', vertical_points, vertical_confidence, LaneType.RIGHT_VERTICAL))
                else:
                    hypotheses.append(LaneHypothesis(x, y, 'horizontal', horizontal_points, horizontal_confidence, LaneType.TOP_HORIZONTAL))
        
        for edge, scan, coord in [('top', top, 0), ('bottom', bottom, h-1), ('left', left, 0), ('right', right, w-1)]:
            peaks, _ = find_peaks(scan, height=min_pixels, distance=margin)
            for peak in peaks:
                if edge in ['top', 'bottom']:
                    add_hypothesis(edge, peak, coord)
                else:
                    add_hypothesis(edge, coord, peak)  # For 'left' and 'right', coord is y and peak is x
        
        
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:max_hypotheses]

    def categorize_hypotheses(self,hypotheses):
        left_vertical = []
        right_vertical = []
        top_horizontal = []

        for hypothesis in hypotheses:
            if hypothesis.lane_type == LaneType.LEFT_VERTICAL:
                left_vertical.append(hypothesis)
                self.left_vertical_status = True
            elif hypothesis.lane_type == LaneType.RIGHT_VERTICAL:
                right_vertical.append(hypothesis)
                self.right_vertical_status = False
            elif hypothesis.lane_type == LaneType.TOP_HORIZONTAL:
                top_horizontal.append(hypothesis)
                self.top_horizontal_status = True

        return left_vertical, right_vertical, top_horizontal

    def sliding_window(self,warped_image, nwindows=9, margin=100, minpix=50, plot=False):
        histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int32(warped_image.shape[0]//nwindows)
        nonzero = warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        if plot:
            out_img = np.dstack((warped_image, warped_image, warped_image)) * 255

        for window in range(nwindows):
            win_y_low = warped_image.shape[0] - (window+1)*window_height
            win_y_high = warped_image.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if plot:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            for i in range(len(ploty)):
                cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 2, (255, 255, 0), -1)
                cv2.circle(out_img, (int(right_fitx[i]), int(ploty[i])), 2, (255, 255, 0), -1)

            cv2.imshow('Sliding Windows', out_img)
            cv2.waitKey(1)

        return left_fit, right_fit

    def sliding_window_top(self, warped_image, nwindows=9, margin=100, minpix=50):
        two_thirds_height = int(2 * warped_image.shape[0] / 3)
        histogram = np.sum(warped_image[:two_thirds_height, :], axis=0)
        
        lane_base = np.argmax(histogram)
        window_width = np.int32(warped_image.shape[1] // nwindows)
        
        nonzero = warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        lane_current = lane_base
        lane_inds = []
        
        out_img_horizontal = np.dstack((warped_image, warped_image, warped_image)) * 255

        for window in range(nwindows):
            win_x_low = window * window_width
            win_x_high = (window + 1) * window_width
            win_y_low = lane_current - margin
            win_y_high = lane_current + margin
        
            cv2.rectangle(out_img_horizontal, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
        
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
            lane_inds.append(good_inds)
        
            if len(good_inds) > minpix:
                lane_current = np.int32(np.mean(nonzeroy[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        lanex = nonzerox[lane_inds]
        laney = nonzeroy[lane_inds]

        if len(lanex) > 0 and len(laney) > 0:
            lane_fit = np.polyfit(lanex, laney, 2)
        else:
            lane_fit = None

        plotx = np.linspace(0, warped_image.shape[1]-1, warped_image.shape[1])
        if lane_fit is not None:
            ploty = lane_fit[0]*plotx**2 + lane_fit[1]*plotx + lane_fit[2]
        else:
            ploty = None

        out_img_horizontal[laney, lanex] = [255, 0, 0]

        return lanex, laney, plotx, ploty, out_img_horizontal, lane_fit

    def apply_sliding_window(self,image, hypotheses):
        left_vertical, right_vertical, top_horizontal = self.categorize_hypotheses(hypotheses)

        lanes = []

        # Apply vertical sliding window for left and right lanes
        if left_vertical or right_vertical:
            left_fit, right_fit = self.sliding_window(image, plot=True)
            if left_fit is not None:
                lanes.append((left_fit, 1.0, LaneType.LEFT_VERTICAL))
            if right_fit is not None:
                lanes.append((right_fit, 1.0, LaneType.RIGHT_VERTICAL))

        # Apply horizontal sliding window for top lanes
        if not lanes and top_horizontal:
            lanex, laney, plotx, ploty, out_img, lane_fit = self.sliding_window_top(image)
            if lane_fit is not None:
                lanes.append((lane_fit, 1.0, LaneType.TOP_HORIZONTAL))

            cv2.imshow('Top Horizontal Lane Detection', out_img)
            cv2.waitKey(1)

        return lanes

    def finding_lane_center(self,warped_image, lanes):
        height, width = warped_image.shape
        overlay_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2RGB)

        if len(lanes) == 0:
            print("No lanes detected")
            return overlay_image, None

        if len(lanes) == 1:
            # Single lane scenario
            lane_fit, _, lane_type = lanes[0]
            
            if lane_type == LaneType.TOP_HORIZONTAL:
                plotx = np.linspace(0, width-1, width)
                lane_fity = lane_fit[0]*plotx**2 + lane_fit[1]*plotx + lane_fit[2]
                
                # Assume lane width is 15cm
                cm_to_pixel_ratio = 10  # Assuming image width is 3 meters
                pixels_to_shift = 15 * cm_to_pixel_ratio
                
                lane_center = lane_fity + pixels_to_shift
                robot_pos = height
                offset = (lane_center[-1] - robot_pos) / (height/2)
                
                # Visualize
                for x, y in zip(plotx, lane_fity):
                    cv2.circle(overlay_image, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # Draw the positive lane center (red)
                for x, y in zip(plotx, lane_center):
                    cv2.circle(overlay_image, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            else:  # LEFT_VERTICAL or RIGHT_VERTICAL
                ploty = np.linspace(0, height-1, height)
                lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
                
                cm_to_pixel_ratio = height / 300  # Assuming image height is 3 meters
                pixels_to_shift = 15 * cm_to_pixel_ratio
                
                if lane_type == LaneType.LEFT_VERTICAL:
                    lane_center = lane_fitx + pixels_to_shift
                else:  # RIGHT_VERTICAL
                    lane_center = lane_fitx - pixels_to_shift
                
                robot_pos = width // 2
                offset = (lane_center[-1] - robot_pos) / (width/2)
                
                # Visualize
                for x, y in zip(lane_center, ploty):
                    cv2.circle(overlay_image, (int(x), int(y)), 2, (0, 255, 0), -1)

        elif len(lanes) == 2:
            # Two lanes scenario
            left_fit, _, left_type = lanes[0] if lanes[0][2] == LaneType.LEFT_VERTICAL else lanes[1]
            right_fit, _, right_type = lanes[1] if lanes[1][2] == LaneType.RIGHT_VERTICAL else lanes[0]
            
            ploty = np.linspace(0, height-1, height)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            lane_center = (left_fitx + right_fitx) // 2
            robot_pos = width // 2
            offset = (lane_center[-1] - robot_pos) / (width/2)
            
            # Visualize
            for y, x in zip(ploty, lane_center):
                cv2.circle(overlay_image, (int(x), int(y)), 2, (0, 255, 0), -1)

        else:
            print(f"Unexpected number of lanes detected: {len(lanes)}")
            return overlay_image, None

        return overlay_image, offset
    
    def move_robot(self,offset,steering):
        twist = Twist()
        twist.linear.x = 0.3  # Constant forward velocity
        twist.angular.z = -offset * steering  # Adjust steering based on offset
        self.cmd_vel_pub.publish(twist)

    def save_image(self, image):
        filename = os.path.join(self.save_folder, f'lane_image_{self.image_counter}.png')
        cv2.imwrite(filename, image)

        self.image_counter += 1

    def save_image(self, image):
        filename = os.path.join(self.save_folder, f'lane_image_{self.image_counter}.png')
        cv2.imwrite(filename, image)
        self.image_counter += 1

    def action_callback(self, msg):
        # Implement your action callback here
        pass
    

def main(args=None):
    rclpy.init(args=args)
    node = RoundaboutNavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()