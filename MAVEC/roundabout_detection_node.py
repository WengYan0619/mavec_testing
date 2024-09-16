import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def perspective_warp(image):
    height, width = image.shape[:2]
    angle = 41
    box_height = 2/3 * height
    deviation = width // 2 - (np.tan(angle * np.pi / 180) * (height - box_height))

    src = np.float32([
        [width // 2 + deviation, box_height],  # top right
        [width // 2 - deviation, box_height],  # top left
        [0, height],  # bottom left
        [width, height]  # bottom right
    ])
    dst = np.float32([
        [width, 0],  # top right
        [0, 0],  # top left
        [0, height],  # bottom left
        [width, height]  # bottom right
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

def threshold(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower = np.array([0, 200, 0])
    upper = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower, upper)
    return white_mask

def sliding_window(warped_image, nwindows=9, margin=100, minpix=50):
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

    out_img = np.dstack((warped_image, warped_image, warped_image)) * 255

    for window in range(nwindows):
        win_y_low = warped_image.shape[0] - (window+1)*window_height
        win_y_high = warped_image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

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

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty, out_img

def sliding_window_top(warped_image, nwindows=9, margin=100, minpix=50):
    # Calculate histogram of top 2/3 of the image
    two_thirds_height = int(2 * warped_image.shape[0] / 3)
    histogram = np.sum(warped_image[:two_thirds_height, :], axis=0)
    print(f"Number of white pixels in each column: {histogram}")

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
    print(f"Lane x: {lanex}, Lane y: {laney}")

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

def move_robot(image, lane_fit):
    height, width = image.shape[:2]
    plotx = np.linspace(0, width-1, width)
    
    # Calculate lane position
    lane_fity = lane_fit[0]*plotx**2 + lane_fit[1]*plotx + lane_fit[2]
    
    # Convert 15cm to pixels
    cm_to_pixel_ratio = 10  # Assuming 1 cm = 10 pixels, adjust this value
    pixels_to_shift = 15 * cm_to_pixel_ratio
    
    # Calculate positive and negative lane centers
    positive_lane_center = lane_fity + pixels_to_shift
    negative_lane_center = lane_fity - pixels_to_shift
    
    # Robot's position (assumed to be at the bottom center of the image)
    robot_pos = height
    
    # Calculate offset
    offset = (np.mean(positive_lane_center) - robot_pos) / (height/2)
    
    print(f"Offset: {offset}")
    
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
    
    return overlay_image, offset

def analyze_white_pixel_orientation(binary_image):
    height, width = binary_image.shape

    # Define regions
    top_region = binary_image[:height//3, :]
    middle_region = binary_image[height//3:2*height//3, :]
    bottom_region = binary_image[2*height//3:, :]
    left_region = binary_image[:, :width//3]
    center_region = binary_image[:, width//3:2*width//3]
    right_region = binary_image[:, 2*width//3:]

    # Count white pixels in each region
    top_count = np.sum(top_region) // 255
    middle_count = np.sum(middle_region) // 255
    bottom_count = np.sum(bottom_region) // 255
    left_count = np.sum(left_region) // 255
    center_count = np.sum(center_region) // 255
    right_count = np.sum(right_region) // 255

    # Calculate vertical and horizontal ratios
    vertical_ratio = (top_count + 1) / (bottom_count + 1)  # Adding 1 to avoid division by zero
    horizontal_ratio = (left_count + right_count + 1) / (center_count + 1)

    # Determine orientation
    if vertical_ratio > 1.5 and horizontal_ratio < 1.5:
        return "top"
    elif vertical_ratio < 0.67 and horizontal_ratio < 1.5:
        return "bottom"
    elif horizontal_ratio > 1.5:
        if left_count > right_count:
            return "left"
        else:
            return "right"
    else:
        return "center"



def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    # Apply perspective warp
    warped = perspective_warp(image)

    # Apply thresholding
    binary = threshold(warped)
    
    white_pixel_position = analyze_white_pixel_orientation(binary)
    print(f"White pixel position: {white_pixel_position}")

    # Apply sliding window for top lane detection
    lanex, laney, plotx, ploty, out_img_horizontal, lane_fit = sliding_window_top(binary)

    if lane_fit is not None:
        height, width = binary.shape[:2]
        plotx = np.linspace(0, width-1, width)
        
        # Calculate lane position
        lane_fity = lane_fit[0]*plotx**2 + lane_fit[1]*plotx + lane_fit[2]

        # Convert 15cm to pixels
        cm_to_pixel_ratio = 10  # Assuming 1 cm = 10 pixels, adjust this value
        pixels_to_shift = 15 * cm_to_pixel_ratio
        
        # Calculate positive and negative lane centers
        positive_lane_center = lane_fity + pixels_to_shift
        negative_lane_center = lane_fity - pixels_to_shift

        # Move robot (simulate movement and get visualization)
        result_image, offset = move_robot(binary, lane_fit)

        # Overlay lane centers on the result image
        for x, y in zip(plotx, lane_fity):
            cv2.circle(result_image, (int(x), int(y)), 2, (0, 255, 0), -1)
        for x, y in zip(plotx, positive_lane_center):
            cv2.circle(result_image, (int(x), int(y)), 2, (0, 0, 255), -1)
        for x, y in zip(plotx, negative_lane_center):
            cv2.circle(result_image, (int(x), int(y)), 2, (255, 0, 0), -1)

        # Display results
        plt.figure(figsize=(20, 10))
        plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
        plt.subplot(222), plt.imshow(binary, cmap='gray'), plt.title('Thresholded Warped Image')
        plt.subplot(223), plt.imshow(out_img_horizontal), plt.title('Sliding Window Result')
        plt.subplot(224), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('Lane Detection Result')
        plt.tight_layout()
        plt.show()
    else:
        print("No lane detected in the image.")

# Main execution
if __name__ == "__main__":
    image_path = r"C:\Users\wengy\Downloads\7.png"  # Replace with your image path
    process_image(image_path)