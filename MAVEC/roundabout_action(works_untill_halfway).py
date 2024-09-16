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
        self.save_folder = os.path.abspath(os.path.join(package_path, '../../../../roundaction_log/',f'roundabout_image_{int(time.time())}'))
        os.mkdir(self.save_folder)
        self.image_counter = 0
        
        #self.frame_buffer = deque(maxlen=1)  # Adjust maxlen as needed
        # Initialize other attributes
        self.turn_frame = None
        self.is_turning = False
        self.turn_complete = False
        self.turning_movement = 0
        self.lane_previous = True
        self.first_frame_number = None

    def warped_image_callback(self, msg):
        if not self.turn_complete and self.turn_frame is None and self.turning_movement <= 10:
            # Capture the first frame for the turn
            try:
                self.turn_frame = self.bridge.imgmsg_to_cv2(msg)
                
                #self.first_frame_number.append(self.turn_frame)
                self.get_logger().info("Turn started. First frame captured.")
                # Timer for processing frames
                if not self.turn_complete:
                    print(f"length of first frame {len(self.turn_frame)}")
                    self.create_timer(3.0, self.process_frame)  # Adjust interval as needed
                
            except Exception as e:
                self.get_logger().error(f'Failed to convert first turn image: {str(e)}')
        elif self.turn_complete:
            # Process frames normally after the turn is complete
            self.lane_previous = False
            self.process_normal_frame(msg)

    def process_frame(self):
        if not self.turn_complete:
            print(f"turn_complete status :{self.turn_complete}")
            self.get_logger().info("First Intialized: Prcoessing the first frame")
            # Process the oldest frame in the buffer
            #frame_to_process = self.frame_buffer[0]
            binary_warped = self.threshold(self.turn_frame)
            lanex, laney, plotx, ploty, out_img_horizontal, lane_fit = self.sliding_window_top(binary_warped)

        
            if lane_fit is not None:
                # Move robot based on detected lane
                self.create_timer(10.0, self.move_robot(binary_warped, lane_fit))
                
                if self.turning_movement >= 2:
                    self.turn_complete = True
                    self.is_turning = False
                    self.turn_frame = None
                
                # # Publish visualization
                # vis_msg = self.bridge.cv2_to_imgmsg(out_img_horizontal, encoding='rgb8')
                # self.image_pub.publish(vis_msg)

                self.get_logger().info("Turn complete. Switching to normal operation.")
                print(f"turn_complete status :{self.turn_complete}")
            else:
                self.get_logger().warn('No lane detected in turn frame')

    def process_normal_frame(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            binary_warped = self.threshold(cv_image)
            left_fit, right_fit = self.sliding_window(binary_warped, plot=False)
            self.move_robot_normal(binary_warped, left_fit, right_fit)
            print(f"turn_complete status: {self.turn_complete}")
            
            # Create visualization
            #out_img = self.visualize_lane(binary_warped, left_fit, right_fit)
            
            # # Publish visualization
            # vis_msg = self.bridge.cv2_to_imgmsg(out_img, encoding='rgb8')
            # self.image_pub.publish(vis_msg)
        except Exception as e:
            self.get_logger().error(f'Error processing normal frame: {str(e)}')
        

    def threshold(self, image):
        # Check if the image is grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            # For grayscale images, just apply a simple threshold
            _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        else:
            # For color images, convert to HLS and then threshold
            hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            lower = np.array([0, 200, 0])
            upper = np.array([255, 255, 255])
            binary = cv2.inRange(hls, lower, upper)
        
        return binary
    
    def sliding_window(self, warped_image, nwindows=9, margin=100, minpix=50, plot=False):
        # Find histogram of bottom half of image
        histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)

        #Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint]) #argmax returns the index of the max value
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint #add midpoint to get the right side

        # Set height of windows
        window_height = np.int(warped_image.shape[0]//nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
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


            # Cycles through the nonzero pixels in the image and checks if they are within the window and appends them to the list
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            #Adds the indices to the list of indices
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            #If the number of pixels found is greater than the minpix, recenter the window
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
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
        # Calculate histogram of top 2/3 of the image
        two_thirds_height = int(2 * warped_image.shape[0] / 3)
        histogram = np.sum(warped_image[:two_thirds_height, :], axis=0)
        #self.get_logger().info(f"Number of white pixels in each column: {histogram}")
    
        # Find peak in the histogram
        lane_base = np.argmax(histogram)
        
        # Set width of windows
        window_width = np.int32(warped_image.shape[1] // nwindows)
    
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        # Current positions to be updated for each window
        lane_current = lane_base
    
        # Create empty lists to receive lane pixel indices
        lane_inds = []
    
        # Create an output image to draw on and visualize the result
        out_img_horizontal = np.dstack((warped_image, warped_image, warped_image)) * 255

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y
            win_x_low = window * window_width
            win_x_high = (window + 1) * window_width
            win_y_low = lane_current - margin
            win_y_high = lane_current + margin
        
            # Draw the windows on the visualization image
            cv2.rectangle(out_img_horizontal, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
        
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
            # Append these indices to the list
            lane_inds.append(good_inds)
        
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                lane_current = np.int32(np.mean(nonzeroy[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract line pixel positions
        lanex = nonzerox[lane_inds]
        laney = nonzeroy[lane_inds]
        #self.get_logger().info(f"Lane x: {lanex}, Lane y: {laney}")

        # Fit a second order polynomial
        if len(lanex) > 0 and len(laney) > 0:
            lane_fit = np.polyfit(lanex, laney, 2)
        else:
            lane_fit = None

        # Generate x and y values for plotting
        plotx = np.linspace(0, warped_image.shape[1]-1, warped_image.shape[1])
        if lane_fit is not None:
            ploty = lane_fit[0]*plotx**2 + lane_fit[1]*plotx + lane_fit[2]
        else:
            ploty = None

        out_img_horizontal[laney, lanex] = [255, 0, 0]

        return lanex, laney, plotx, ploty, out_img_horizontal, lane_fit

    def move_robot(self, image, lane_fit):
        
        self.turning_movement +=1
        height, width = image.shape[:2]
        plotx = np.linspace(0, width-1, width)
        
        # Calculate lane position
        lane_fity = lane_fit[0]*plotx**2 + lane_fit[1]*plotx + lane_fit[2]
        
        # Convert 15cm to pixels
        # You need to know the conversion factor between cm and pixels in your image
        cm_to_pixel_ratio = 10  # Assuming 1 cm = 10 pixels, adjust this value
        pixels_to_shift = 15 * cm_to_pixel_ratio
        
        # Calculate positive and negative lane centers
        positive_lane_center = lane_fity + pixels_to_shift
        negative_lane_center = lane_fity - pixels_to_shift
        
        # Robot's position (assumed to be at the bottom center of the image)
        robot_pos = height
        
        # Calculate offset
        offset = (positive_lane_center[-1] - robot_pos) / (height/2)
        
        if self.lane_previous == True and not self.turn_complete:
            offset_previous = offset

            # Create and publish Twist message
            twist = Twist()
            twist.linear.x = 0.3  # Constant forward velocity
            twist.angular.z = -offset_previous * 2.0  # Adjust steering based on offset
            self.cmd_vel_pub.publish(twist)
            
        
        else:
            # Create and publish Twist message
            twist = Twist()
            twist.linear.x = 0.3  # Constant forward velocity
            twist.angular.z = -offset * 2.0  # Adjust steering based on offset
            self.cmd_vel_pub.publish(twist)
            
        
        self.get_logger().info(f"Offset: {offset}, Linear: {twist.linear.x}, Angular: {twist.angular.z}")
        
       # Overlay lane center line on the image
        overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for x, y in zip(plotx, lane_fity):
            cv2.circle(overlay_image, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # Draw the positive lane center (red)
        for x, y in zip(plotx, positive_lane_center):
            cv2.circle(overlay_image, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Draw the negative lane center (blue)
        for x, y in zip(plotx, negative_lane_center):
            cv2.circle(overlay_image, (int(x), int(y)), 2, (255, 0, 0), -1)
        
        # Display the image with the lane center line
        cv2.imshow('Lane Center', overlay_image)
        cv2.waitKey(1)
        
        self.save_image(overlay_image)
        
    def move_robot_normal(self, image, left_fit, right_fit):
        height, width = image.shape[:2]

        ploty = np.linspace(0, height-1, height)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        lane_center = (left_fitx + right_fitx) // 2
        robot_pos = width // 2

        offset = (lane_center[-1] - robot_pos) / (width/2)

        twist = Twist()
        twist.linear.x = 0.2
        twist.angular.z = -offset * 3.0
        self.cmd_vel_pub.publish(twist)

        print(f"Offset: {offset}, Linear: {twist.linear.x}, Angular: {twist.angular.z}")

        # Overlay lane center line on the image
        overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for y, x in zip(ploty, lane_center):
            cv2.circle(overlay_image, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Display the image with the lane center line
        #cv2.imshow('Lane Center', overlay_image)
        cv2.waitKey(1)  

        self.save_image(overlay_image)

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