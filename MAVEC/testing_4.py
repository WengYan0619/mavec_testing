
import numpy as np
import cv2
from scipy.signal import find_peaks
from dataclasses import dataclass
from enum import Enum

class LaneType(Enum):
    LEFT_TOP = 1
    RIGHT_TOP = 2
    TOP_HORIZONTAL = 3
    BOTTOM_LEFT = 4
    BOTTOM_RIGHT = 5

@dataclass
class LaneHypothesis:
    start_x: int
    start_y: int
    direction: str
    traced_points: list
    confidence: float
    lane_type: LaneType
    
def threshold(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        lower = np.array([0, 200, 0])
        upper = np.array([255, 255, 255])
        white_mask = cv2.inRange(hls, lower, upper)

        return white_mask
    
    else:
        return cv2.inRange(image, 200,255)
    
def perspective_warp(image, box_height, angle=41):
    height, width = image.shape[:2]

    deviation = width//2 - (np.tan(angle*np.pi/180) * (height - box_height))

    # print(f"BH: {box_height}, Deviation: {deviation}")

    src = np.float32([
        [width//2 + deviation, box_height], #top right
        [width//2 - deviation, box_height], #top left
        [0, height], #bottom left
        [width, height] #bottom right
    ])

    dst = np.float32([
        [width, 0], #top right
        [0, 0], #top left
        [0, height], #bottom left
        [width, height] #bottom right
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    # polyline = cv2.polylines(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), [np.int32(src)], isClosed=True, color=(0, 255, 0), thickness=3)
    # cv2.imshow("Polyline", polyline)
    # cv2.waitKey(1)

    return warped


def detect_initial_lanes(image, margin=20, min_pixels=50, trace_length=40, max_hypotheses=5):
    h, w = image.shape
    
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
            continuity_score = 0.0
            print("There is no continuity score for this edge")
        
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
                add_hypothesis(edge, coord, peak)
    
    hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    return hypotheses[:max_hypotheses]

def categorize_hypotheses(hypotheses):
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

def sliding_window(warped_image, nwindows=9, margin=100, minpix=50, plot=False):
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

def sliding_window_top(warped_image, nwindows=9, margin=100, minpix=50):
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

def apply_sliding_window(image, hypotheses):
    left_vertical, right_vertical, top_horizontal = categorize_hypotheses(hypotheses)

    lanes = []

    # Apply vertical sliding window for left and right lanes
    if left_vertical or right_vertical:
        left_fit, right_fit = sliding_window(image, plot=True)
        if left_fit is not None:
            lanes.append((left_fit, 1.0, LaneType.LEFT_VERTICAL))
        if right_fit is not None:
            lanes.append((right_fit, 1.0, LaneType.RIGHT_VERTICAL))

    # Apply horizontal sliding window for top lanes
    if not lanes and top_horizontal:
        lanex, laney, plotx, ploty, out_img, lane_fit = sliding_window_top(image)
        if lane_fit is not None:
            lanes.append((lane_fit, 1.0, LaneType.TOP_HORIZONTAL))

        cv2.imshow('Top Horizontal Lane Detection', out_img)
        cv2.waitKey(1)

    return lanes

def finding_lane_center(warped_image, lanes):
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

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholding_image = threshold(image)
    warped_image = perspective_warp(thresholding_image, 2/3*image.shape[0])
    
    hypotheses = detect_initial_lanes(warped_image)
    print(f"Number of hypotheses: {len(hypotheses)}")
    for h in hypotheses:
        print(f"Hypothesis: {h.lane_type}, confidence: {h.confidence}")
    
    lanes = apply_sliding_window(warped_image, hypotheses)
    print(f"Number of lanes detected: {len(lanes)}")
    for lane in lanes:
        print(f"Lane type: {lane[2]}")
    
    overlay_image, offset = finding_lane_center(warped_image, lanes)
    
    if overlay_image is not None:
        cv2.imshow('Lane Center', overlay_image)
        if offset is not None:
            print(f"Offset: {offset}")
        else:
            print("No offset calculated")
    else:
        print("No valid overlay image generated")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return lanes

# Usage
lanes = process_image(r"C:\Users\wengy\Downloads\13.png")