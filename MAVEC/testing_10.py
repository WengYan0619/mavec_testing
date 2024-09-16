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

def sliding_window(warped_image, nwindows=15, margin=150, minpix=200,plot = True):
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


def detect_white_pixels_and_fit_polynomial(image, degree=2, plot=True):
    # Convert image to binary if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    else:
        binary = image

    # Find white pixels
    white_pixels = np.where(binary == 255)
    white_pixel_count = len(white_pixels[0])

    # Fit polynomial
    y_pixels = white_pixels[0]
    x_pixels = white_pixels[1]
    poly_coeffs = np.polyfit(y_pixels, x_pixels, degree)
    poly_func = np.poly1d(poly_coeffs)

    # Generate points for plotting the polynomial
    plot_y = np.linspace(0, binary.shape[0]-1, binary.shape[0])
    plot_x = poly_func(plot_y)

    if plot:
        # Create an output image to draw on
        out_img = np.dstack((binary, binary, binary))

        # Plot white pixels
        out_img[white_pixels] = [255, 0, 0]  # Mark white pixels in red

        # Plot polynomial line
        for i in range(len(plot_y)):
            if 0 <= plot_x[i] < binary.shape[1]:  # Ensure point is within image bounds
                cv2.circle(out_img, (int(plot_x[i]), int(plot_y[i])), 2, (0, 255, 0), -1)

        # Add text for white pixel count
        cv2.putText(out_img, f'White Pixels: {white_pixel_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('White Pixel Detection with Polynomial Fit', out_img)
        cv2.waitKey(1)

    return white_pixels, white_pixel_count, poly_coeffs, out_img if plot else None

# Example usage:
# white_pixels, count, poly_coeffs, output_image = detect_white_pixels_and_fit_polynomial(your_image)


def sliding_window_single_lane(warped_image, nwindows=15, margin=300, minpix=200, plot=True):
    # Find all white pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_image, warped_image, warped_image)) * 255
    
    # Find the peak of the sum of the bottom half of the image
    histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    base = np.argmax(histogram)
    print(f"Base: {base}")
    print(f"Histogram: {histogram}")
    # Set height of windows
    window_height = np.int32(warped_image.shape[0] // nwindows)
    
    # Current position to be updated later for each window in nwindows
    current_x = base
    
    # Create empty lists to receive lane pixel indices
    lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_image.shape[0] - (window + 1) * window_height
        win_y_high = warped_image.shape[0] - window * window_height
        win_x_low = current_x - margin
        win_x_high = current_x + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
        # Append these indices to the lists
        lane_inds.append(good_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            current_x = np.int32(np.mean(nonzerox[good_inds]))
    
    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)
    
    # Extract lane line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]
    
    # Fit a second order polynomial
    fit = np.polyfit(y, x, 2)
    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    
    # Generate x and y values for plotting
    out_img[y, x] = [255, 0, 0]
    
    if plot:
        for i in range(len(ploty)):
            cv2.circle(out_img, (int(fitx[i]), int(ploty[i])), 2, (0, 255, 255), -1)
        cv2.imshow('Sliding Window Single Lane', out_img)
        cv2.waitKey(1)
    
    return x, y, fitx, ploty, out_img

# Example usage:
# x, y, fitx, ploty, out_img = sliding_window_single_lane(warped_image)

import numpy as np
import cv2

def sliding_window_flexible_lane(warped_image, nwindows=15, margin=100, minpix=50, plot=True):
    # Find all white pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_image, warped_image, warped_image)) * 255
    
    # Determine if the lane is more horizontal or vertical
    vertical_histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    horizontal_histogram = np.sum(warped_image[:,:warped_image.shape[1]//2], axis=1)
    
    is_vertical = np.max(vertical_histogram) > np.max(horizontal_histogram)
    
    if is_vertical:
        base = np.argmax(vertical_histogram)
        window_height = np.int32(warped_image.shape[0] // nwindows)
        current_x = base
    else:
        base = np.argmax(horizontal_histogram)
        window_width = np.int32(warped_image.shape[1] // nwindows)
        current_y = base
    
    # Create empty lists to receive lane pixel indices
    lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        if is_vertical:
            win_y_low = warped_image.shape[0] - (window + 1) * window_height
            win_y_high = warped_image.shape[0] - window * window_height
            win_x_low = current_x - margin
            win_x_high = current_x + margin
        else:
            win_x_low = window * window_width
            win_x_high = (window + 1) * window_width
            win_y_low = current_y - margin
            win_y_high = current_y + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
        # Append these indices to the lists
        lane_inds.append(good_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            if is_vertical:
                current_x = np.int32(np.mean(nonzerox[good_inds]))
            else:
                current_y = np.int32(np.mean(nonzeroy[good_inds]))
    
    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)
    
    # Extract lane line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]
    
    # Convert 15cm to pixels
    cm_to_pixel_ratio = 10  # Assuming 1 cm = 10 pixels, adjust this value
    pixels_to_shift = 20 * cm_to_pixel_ratio
    
    # Fit a second order polynomial
    if is_vertical:
        fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        plot_points = list(zip(fitx, ploty))
        
        # Calculate positive and negative lane centers
        positive_lane_center = fitx + pixels_to_shift
        negative_lane_center = fitx - pixels_to_shift
        
    else:
        fit = np.polyfit(x, y, 2)
        plotx = np.linspace(0, warped_image.shape[1]-1, warped_image.shape[1])
        fity = fit[0]*plotx**2 + fit[1]*plotx + fit[2]
        plot_points = list(zip(plotx, fity))
        
        # Calculate positive and negative lane centers
        positive_lane_center = fity + pixels_to_shift
        negative_lane_center = fity - pixels_to_shift
    
    # Generate x and y values for plotting
    out_img[y, x] = [255, 0, 0]
    
    if plot:
        for point in plot_points:
            x, y = map(int, point)
            if 0 <= x < warped_image.shape[1] and 0 <= y < warped_image.shape[0]:
                cv2.circle(out_img, (x, y), 2, (0, 255, 255), -1)
        cv2.imshow('Sliding Window Flexible Lane', out_img)
        cv2.waitKey(1)
        
    cv2.imshow('Sliding Window Flexible Lane', out_img)
    cv2.waitKey(1)
    
    
    return x, y, plot_points, out_img,positive_lane_center,negative_lane_center,is_vertical



def finding_lane_center(warped_image, positive_lane_center, negative_lane_center,is_vertical):
    height, width = warped_image.shape
    overlay_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2RGB)
    robot_pos = height  # Assuming robot is at the bottom of the image

    # Initialize arrays to store distances
    positive_distances = []
    negative_distances = []

    # Calculate Euclidean distances for each point along the lane
    for i in range(len(positive_lane_center)):
        if not is_vertical:  # Horizontal lane
            pos_dist = np.sqrt((i - robot_pos)**2 + (positive_lane_center[i] - width/2)**2)
            neg_dist = np.sqrt((i - robot_pos)**2 + (negative_lane_center[i] - width/2)**2)
        else:  # Vertical lane
            pos_dist = np.sqrt((positive_lane_center[i] - width/2)**2 + (i - robot_pos)**2)
            neg_dist = np.sqrt((negative_lane_center[i] - width/2)**2 + (i - robot_pos)**2)
        
        positive_distances.append(pos_dist)
        negative_distances.append(neg_dist)

    # Find the minimum distance and corresponding index
    min_pos_dist = min(positive_distances)
    min_neg_dist = min(negative_distances)

    if min_pos_dist < min_neg_dist:
        
        chosen_center = positive_lane_center
        #min_dist_index = positive_distances.index(min_pos_dist)
        #offset = chosen_center[min_dist_index] - width/2
        print("The chosen center is the positive lane center")
    else:
        chosen_center = negative_lane_center
        #min_dist_index = negative_distances.index(min_neg_dist)
        #offset = chosen_center[min_dist_index] - width/2
        print("the chosen center is the negative lane center")
    
    #print(f"Chosen center: {chosen_center}")
    if is_vertical:
        offset = (chosen_center[-1] - robot_pos) / (width/2)
    else:
        offset = (chosen_center[-1] - robot_pos) / (height/2)

    if is_vertical:
        # Draw both lane centers
        # for i, center in enumerate(zip(positive_lane_center, negative_lane_center)):
        #     cv2.circle(overlay_image, (int(center[0]), i), 1, (0, 255, 255), -1)  # Yellow
        #     cv2.circle(overlay_image, (int(center[1]), i), 1, (255, 0, 255), -1)  # Magenta

        # Draw chosen center
        for i, point in enumerate(chosen_center):
            cv2.circle(overlay_image, (int(point), i), 2, (0, 255, 0), -1)  # Green

        # Draw robot position
        # cv2.circle(overlay_image, (width//2, robot_pos), 5, (255, 0, 0), -1)  # Blue

        # Draw line from robot to chosen center
        #cv2.line(overlay_image, (width//2, robot_pos), (int(chosen_center[-1]), robot_pos), (0, 0, 255), 2)  # Red

    else:  # Horizontal lane
        # Draw both lane centers
        # for i, center in enumerate(zip(positive_lane_center, negative_lane_center)):
        #     cv2.circle(overlay_image, (i, int(center[0])), 1, (0, 255, 255), -1)  # Yellow
        #     cv2.circle(overlay_image, (i, int(center[1])), 1, (255, 0, 255), -1)  # Magenta

        # Draw chosen center
        for i, point in enumerate(chosen_center):
            cv2.circle(overlay_image, (i, int(point)), 2, (0, 255, 0), -1)  # Green

        # Draw robot position
        cv2.circle(overlay_image, (width-1, height//2), 5, (255, 0, 0), -1)  # Blue

        # Draw line from robot to chosen center
        #cv2.line(overlay_image, (width-1, height//2), (width-1, int(chosen_center[-1])), (0, 0, 255), 2)  # Red

    # Add text for offset
    cv2.putText(overlay_image, f'Offset: {offset:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Lane Center Detection', overlay_image)
    cv2.waitKey(1)

    return offset, overlay_image

# Example usage:
# offset, visualized_image = finding_lane_center(warped_image, positive_lane_center, negative_lane_center)
# Example usage:
# x, y, plot_points, out_img = sliding_window_flexible_lane(warped_image)
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    exit_condition = 0
    if exit_condition == 0:
        height, width = image.shape[:2]
        
        # Create a mask for the trapezoidal cut-off
        mask = np.ones_like(image, dtype=np.uint8)
        
        # Calculate start points for top and bottom
        top_start = int(2.5 * width // 4)
        bottom_start = int(3 * width // 4)
        
        # Create a gradient for the cut-off line
        x = np.arange(height)
        cut_off_line = np.int32(np.interp(x, [0, height-1], [top_start, bottom_start]))
        
        # Apply the mask
        for i in range(height):
            mask[i, cut_off_line[i]:] = 0
        
        # Apply the mask to the current frame
        image = image * mask
    # Apply perspective warp
    warped = perspective_warp(image)

    # Apply thresholding
    binary = threshold(warped)
    #white_pixels, count, poly_coeffs, output_image = detect_white_pixels_and_fit_polynomial(binary)
    #x, y, fitx, ploty, out_img = sliding_window_single_lane(binary)
    #white_pixel_position = analyze_white_pixel_orientation(binary)
    x, y, plot_points, out_img,positive_lane_center,negative_lane_center,is_vertical = sliding_window_flexible_lane(binary)
    offset, visualized_image = finding_lane_center(binary, positive_lane_center, negative_lane_center, is_vertical)
    #print(f"White pixel position: {white_pixel_position}")
    overlay_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    # Apply sliding window for top lane detection
    #lanex, laney, plotx, ploty, out_img_horizontal, lane_fit = sliding_window_top(binary)
    #leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty, out_img = sliding_window(binary)

    
    # height, width = binary.shape[:2]
    # lane_center = (left_fitx + right_fitx) // 2
    # robot_pos = width // 2
    # offset = (lane_center[-1] - robot_pos) / (width/2)
    
    # # Visualize
    # # for y, x in zip(ploty, lane_center):
    # #     cv2.circle(overlay_image, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    # if lane_fit is not None:
    #     height, width = binary.shape[:2]
    #     plotx = np.linspace(0, width-1, width)
        
    #     # Calculate lane position
    #     lane_fity = lane_fit[0]*plotx**2 + lane_fit[1]*plotx + lane_fit[2]

    #     # Convert 15cm to pixels
    #     cm_to_pixel_ratio = 10  # Assuming 1 cm = 10 pixels, adjust this value
    #     pixels_to_shift = 15 * cm_to_pixel_ratio
        
    #     # Calculate positive and negative lane centers
    #     positive_lane_center = lane_fity + pixels_to_shift
    #     negative_lane_center = lane_fity - pixels_to_shift

    #     # Move robot (simulate movement and get visualization)
    #     result_image, offset = move_robot(binary, lane_fit)

    #     # Overlay lane centers on the result image
    #     for x, y in zip(plotx, lane_fity):
    #         cv2.circle(result_image, (int(x), int(y)), 2, (0, 255, 0), -1)
    #     for x, y in zip(plotx, positive_lane_center):
    #         cv2.circle(result_image, (int(x), int(y)), 2, (0, 0, 255), -1)
    #     for x, y in zip(plotx, negative_lane_center):
    #         cv2.circle(result_image, (int(x), int(y)), 2, (255, 0, 0), -1)

    #     # Display results
    #     plt.figure(figsize=(20, 10))
    #     plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    #     plt.subplot(222), plt.imshow(binary, cmap='gray'), plt.title('Thresholded Warped Image')
    #     plt.subplot(223), plt.imshow(out_img_horizontal), plt.title('Sliding Window Result')
    #     plt.subplot(224), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('Lane Detection Result')
    #     plt.tight_layout()
    #     plt.show()
        
    cv2.imshow("Overlay Image", overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    image_path = r"C:\Users\wengy\Downloads\7.png"  # Replace with your image path
    process_image(image_path)