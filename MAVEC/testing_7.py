import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray, Float64
import os
from ament_index_python.packages import get_package_share_directory
import time
from enum import Enum
from dataclasses import dataclass
from scipy.signal import find_peaks  # Add this line

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
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 10)

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

        # Moving average attributes
        self.mov_avg_left = np.array([])
        self.mov_avg_right = np.array([])
        self.mov_avg_top = np.array([])
        self.left_fit = None
        self.right_fit = None
        self.top_fit = None

    def warped_image_callback(self, msg):
        current_frame = self.bridge.imgmsg_to_cv2(msg)
        
        if self.top_horizontal_status and self.turn_frame is None and not self.turn_complete:
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
        overlay_image, offset = self.make_lane(frame_to_process, lanes)
        
        # Publish the visualization
        if overlay_image is not None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(overlay_image, "bgr8"))
        
        # Save the image
        self.save_image(overlay_image)
        
        # Use the offset for navigation
        if offset is not None:
            if self.turn_frame is not None and not self.turn_complete:
                self.move_robot(offset, 3.0)
                self.turning_movement += 1
                if self.turning_movement >= 4:
                    self.turn_complete = True
                    self.turn_frame = None
                    self.top_horizontal_status = False
                    self.get_logger().info("Turn complete. Switching to normal operation.")
            else:
                self.move_robot(offset, 4.0)

    def detect_initial_lanes(self, image, margin=20, min_pixels=50, trace_length=40, max_hypotheses=5):
        if image is None:
            self.get_logger().error("Received None image in detect_initial_lanes")
            return []
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

    def categorize_hypotheses(self, hypotheses):
        left_vertical = []
        right_vertical = []
        top_horizontal = []

        for hypothesis in hypotheses:
            if hypothesis.lane_type == LaneType.LEFT_VERTICAL:
                left_vertical.append(hypothesis)
            elif hypothesis.lane_type == LaneType.RIGHT_VERTICAL:
                right_vertical.append(hypothesis)
            elif hypothesis.lane_type == LaneType.TOP_HORIZONTAL:
                top_horizontal.append(hypothesis)

        return left_vertical, right_vertical, top_horizontal

    def sliding_window(self, warped_image, nwindows=9, margin=100, minpix=50, plot=False):
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

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

        if plot:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
            if left_fit is not None:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                for i in range(len(ploty)):
                    cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 2, (255, 255, 0), -1)
            if right_fit is not None:
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                for i in range(len(ploty)):
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

    def apply_sliding_window(self, image, hypotheses):
        left_vertical, right_vertical, top_horizontal = self.categorize_hypotheses(hypotheses)
        lanes = []
        MOV_AVG_LENGTH = 5

        # Apply vertical sliding window for left and right lanes
        if left_vertical or right_vertical:
            left_fit, right_fit = self.sliding_window(image, plot=True)
            if left_fit is not None:
                self.mov_avg_left = np.append(self.mov_avg_left, np.array([left_fit]), axis=0)
                lanes.append((left_fit, 1.0, LaneType.LEFT_VERTICAL))
                self.left_vertical_status = True
            if right_fit is not None:
                self.mov_avg_right = np.append(self.mov_avg_right, np.array([right_fit]), axis=0)
                lanes.append((right_fit, 1.0, LaneType.RIGHT_VERTICAL))
                self.right_vertical_status = True

            # Calculate moving averages for vertical lanes
            if self.mov_avg_left.shape[0] > 0:
                self.left_fit = np.array([np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
                                          np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
                                          np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])])
            
            if self.mov_avg_right.shape[0] > 0:
                self.right_fit = np.array([np.mean(self.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
                                           np.mean(self.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
                                           np.mean(self.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])])

            # Memory management for vertical lanes
            if self.mov_avg_left.shape[0] > 1000:
                self.mov_avg_left = self.mov_avg_left[-MOV_AVG_LENGTH:]
            if self.mov_avg_right.shape[0] > 1000:
                self.mov_avg_right = self.mov_avg_right[-MOV_AVG_LENGTH:]

        # Apply horizontal sliding window for top lanes
        if top_horizontal:
            lanex, laney, plotx, ploty, out_img, lane_fit = self.sliding_window_top(image)
            if lane_fit is not None:
                self.mov_avg_top = np.append(self.mov_avg_top, np.array([lane_fit]), axis=0)
                lanes.append((lane_fit, 1.0, LaneType.TOP_HORIZONTAL))
                self.top_horizontal_status = True

                # Calculate moving average for top horizontal lane
                if self.mov_avg_top.shape[0] > 0:
                    self.top_fit = np.array([np.mean(self.mov_avg_top[::-1][:, 0][0:MOV_AVG_LENGTH]),
                                             np.mean(self.mov_avg_top[::-1][:, 1][0:MOV_AVG_LENGTH]),
                                             np.mean(self.mov_avg_top[::-1][:, 2][0:MOV_AVG_LENGTH])])

                # Memory management for top horizontal lane
                if self.mov_avg_top.shape[0] > 1000:
                    self.mov_avg_top = self.mov_avg_top[-MOV_AVG_LENGTH:]

            cv2.imshow('Top Horizontal Lane Detection', out_img)
            cv2.waitKey(1)

        return lanes

    def make_lane(self, cv_image, lanes):
        height, width = cv_image.shape[:2]
        warp_zero = np.zeros((height, width, 1), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
        
        ploty = np.linspace(0, height - 1, height)
        self.is_center_x_exist = False
        centerx = None

        for lane_fit, confidence, lane_type in lanes:
            if lane_type in [LaneType.LEFT_VERTICAL, LaneType.RIGHT_VERTICAL]:
                fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
                pts = np.array([np.transpose(np.vstack([fitx, ploty]))])
                color = (0, 0, 255) if lane_type == LaneType.LEFT_VERTICAL else (255, 255, 0)
                cv2.polylines(color_warp_lines, np.int_([pts]), isClosed=False, color=color, thickness=25)
            elif lane_type == LaneType.TOP_HORIZONTAL:
                plotx = np.linspace(0, width - 1, width)
                fity = lane_fit[0]*plotx**2 + lane_fit[1]*plotx + lane_fit[2]
                pts = np.array([np.transpose(np.vstack([plotx, fity]))])
                cv2.polylines(color_warp_lines, np.int_([pts]), isClosed=False, color=(0, 255, 255), thickness=25)

        if len(lanes) == 2 and all(lane[2] in [LaneType.LEFT_VERTICAL, LaneType.RIGHT_VERTICAL] for lane in lanes):
            left_fitx = lanes[0][0][0]*ploty**2 + lanes[0][0][1]*ploty + lanes[0][0][2]
            right_fitx = lanes[1][0][0]*ploty**2 + lanes[1][0][1]*ploty + lanes[1][0][2]
            centerx = np.mean([left_fitx, right_fitx], axis=0)
            pts = np.hstack((np.array([np.transpose(np.vstack([left_fitx, ploty]))]),
                             np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])))
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            self.is_center_x_exist = True
        elif len(lanes) == 1:
            lane_fit, _, lane_type = lanes[0]
            if lane_type == LaneType.LEFT_VERTICAL:
                fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
                centerx = fitx + 320  # Assuming 320 is half the lane width
            elif lane_type == LaneType.RIGHT_VERTICAL:
                fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
                centerx = fitx - 320
            elif lane_type == LaneType.TOP_HORIZONTAL:
                centerx = np.full_like(ploty, width // 2)  # Assume center is middle of image
            self.is_center_x_exist = True

        if centerx is not None:
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            cv2.polylines(color_warp_lines, np.int_([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)

        final = cv2.addWeighted(cv_image, 1, color_warp, 0.2, 0)
        final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)

        offset = None
        if self.is_center_x_exist:
            center_offset = centerx[350] - (width // 2)  # Offset at the bottom of the image
            offset = center_offset / (width // 2)  # Normalize offset
            msg_desired_center = Float64()
            msg_desired_center.data = centerx[350]
            self.pub_lane.publish(msg_desired_center)

        return final, offset

    def move_robot(self, offset, steering):
        twist = Twist()
        twist.linear.x = 0.3  # Constant forward velocity
        twist.angular.z = -offset * steering  # Adjust steering based on offset
        self.cmd_vel_pub.publish(twist)
        
        print(f"Linear Velocity: {twist.linear.x}")
        print(f"Angular Velocity: {twist.angular.z}")

    def save_image(self, image):
        filename = os.path.join(self.save_folder, f'lane_image_{self.image_counter}.png')
        cv2.imwrite(filename, image)
        self.image_counter += 1

    def action_callback(self, msg):
        # Implement your action callback here
        pass

    def main(self):
        rclpy.spin(self)

def main(args=None):
    rclpy.init(args=args)
    node = RoundaboutNavigationNode()
    try:
        node.main()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()