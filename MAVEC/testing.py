import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load and process the image
image_path = r"C:\Users\wengy\Downloads\round 11.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Warp the binary image
height, width = gray.shape[:2]
angle = 41
box_height = 2/3 * gray.shape[0]
deviation = width//2 - (np.tan(angle*np.pi/180) * (height - box_height))
src = np.float32([
    [width//2 + deviation, box_height],
    [width//2 - deviation, box_height],
    [0, height],
    [width, height]
])
dst = np.float32([
    [width, 0],
    [0, 0],
    [0, height],
    [width, height]
])
matrix = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(gray, matrix, (width, height))

def analyze_lane_visibility(warped_image):
    warped_height, warped_width = warped_image.shape[:2]
   
    # Define the region of interest (ROI) for lane visibility  
    top_half = warped_image[:warped_height//2, :]
    bottom_half = warped_image[warped_height//2:, :]
   
    # Calculate the percentage of white pixels in the top and bottom halves
    ratio_threshold = 1
    top_half_white_pixels = np.sum(top_half > 200)
    bottom_half_white_pixels = np.sum(bottom_half > 200)
   
    #print(f"Unique values in warped_image: {np.unique(warped_image)}")
    #print(f"Max value in warped_image: {np.max(warped_image)}")
    print(f"White pixels top: {top_half_white_pixels}, white pixels bottom: {bottom_half_white_pixels}")
   
    if top_half_white_pixels == bottom_half_white_pixels :
        return False
    # Avoid division by zero
    elif bottom_half_white_pixels == 0:
        bottom_half_white_pixels = top_half_white_pixels
        print("Bottom half has no white pixels, so its not set to the same value")
        #return False
    elif top_half_white_pixels == 0:
        top_half_white_pixels = bottom_half_white_pixels
        print("Top half has no white pixels, so its now set to the same value")
        #return False
   
    ratio = top_half_white_pixels / bottom_half_white_pixels
    print(f"Ratio: {ratio}")
   
    return ratio <= ratio_threshold

def move_robot(self, image, left_fit):
    height, width = image.shape[:2]
    ploty = np.linspace(0, height-1, height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    
    # Estimate the right lane based on the left lane
    right_fitx = left_fitx + self.lane_width_pixels
    
    lane_center = (left_fitx + right_fitx) // 2
    robot_pos = width // 2
    offset = (lane_center[-1] - robot_pos) / (width/2)
    
    # Adjust the speed and turning rate for the small scale
    max_speed = 0.05  # m/s, adjust based on your robot's capabilities
    max_turn_rate = 1.0  # rad/s, adjust based on your robot's capabilities

    twist = Twist()
    twist.linear.x = max_speed * (1 - abs(offset))  # Slow down when off-center
    twist.angular.z = -np.clip(offset * max_turn_rate, -max_turn_rate, max_turn_rate)
    self.publisher.publish(twist)
    
    self.get_logger().info(f"Offset: {offset:.4f}, Linear: {twist.linear.x:.4f}, Angular: {twist.angular.z:.4f}")
    
    # Overlay lane center line on the image
    overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for y, x in zip(ploty, lane_center):
        cv2.circle(overlay_image, (int(x), int(y)), 1, (0, 255, 0), -1)
    
    # Draw left and right lane lines
    for y, x in zip(ploty, left_fitx):
        cv2.circle(overlay_image, (int(x), int(y)), 1, (255, 0, 0), -1)
    for y, x in zip(ploty, right_fitx):
        cv2.circle(overlay_image, (int(x), int(y)), 1, (0, 0, 255), -1)
    
    # Publish the visualization
    self.image_pub.publish(self.bridge.cv2_to_imgmsg(overlay_image, encoding="rgb8"))
    
    # Save image periodically
    if self.image_counter % 10 == 0:  # Save every 10th frame
        self.save_image(overlay_image)
    self.image_counter += 1

def save_image(self, image):
    filename = os.path.join(self.save_folder, f'lane_image_{self.image_counter:04d}.png')
    cv2.imwrite(filename, image)


def main(warped):
    self.left_fit = self.sliding_window(warped)
    is_circle_visible = True
    iteration = 0
    while is_circle_visible and iteration < 5:
        circle_visible = analyze_lane_visibility(warped)
        
        if circle_visible:
            print("Circle is visible")
            move_robot()
        else:
            print("Circle is not visible")
            is_circle_visible = False
            break
            print("Now turn left and perform sliding window to find the lane, we focus on the left side of the image")
            
        iteration += 1
    move_robot(warped,left_fit)
    # Display results
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(warped, cmap='gray')
    plt.title('Warped Image')
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title('Original Image')
    plt.show()

# Call the main function
main(warped)