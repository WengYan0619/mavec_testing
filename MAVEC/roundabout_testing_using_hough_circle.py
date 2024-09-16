import cv2
import numpy as np
from scipy.spatial import distance

def perspective_warp(image):
    height, width = image.shape[:2]
    angle = 41
    box_height = 2/3 * image.shape[0]
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

def preprocess_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_arcs(image, min_radius=100, max_radius=200):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=0.3, minDist=80,
                               param1=50, param2=13, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"Detected {circles} circles/arcs")
        print(f"Detected {len(circles)} circles/arcs")
    else:
        print("No circles/arcs detected")
        circles = []
    return circles

def find_center(circles):
    resulted_image = np.zeros((512, 512, 3), dtype=np.uint8)
    centers = circles[:, :2]
    radii = circles[:, 2]
    threshold = 30
    # List to store the new circles (mean center and radius)
    mean_circles = []
    visited = set()
    # Find groups of close centers and calculate their mean center and mean radius
    for i in range(len(centers)):
        if i in visited:
            continue
        
        cluster = [i]
        for j in range(i + 1, len(centers)):
            if j in visited:
                continue
            
            dist = distance.euclidean(centers[i], centers[j])
            if dist < threshold:
                cluster.append(j)
                visited.add(j)
        
        # Calculate the mean center and mean radius for the cluster
        mean_center = np.mean(centers[cluster], axis=0)
        mean_radius = np.mean(radii[cluster])
        
        mean_circles.append((mean_center, mean_radius))

        # Draw the big circle with the mean center and radius
        mean_center = tuple(mean_center.astype(int))
        mean_radius = int(mean_radius)
        cv2.circle(resulted_image, mean_center, mean_radius, (0, 255, 0), 2)  # Green circle
        
        return resulted_image

def create_detailed_heatmap(image, circles):
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    for (x, y, r) in circles:
        cv2.circle(heatmap, (x, y), r, 1, thickness=2)
    return heatmap

def isolate_prominent_arc(heatmap, threshold_ratio=0.5):
    # Normalize heatmap
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply adaptive thresholding
    max_value = np.max(heatmap_normalized)
    threshold = int(max_value * threshold_ratio)
    _, binary = cv2.threshold(heatmap_normalized, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the contour with the highest average intensity in the heatmap
        best_contour = max(contours, key=lambda c: np.mean(heatmap[cv2.drawContours(np.zeros_like(binary), [c], 0, 1, 1)]))
        return best_contour
    return None

def main():
    image_path = r"C:\Users\wengy\Downloads\round 11.png"
    original_image = cv2.imread(image_path)
   
    if original_image is None:
        print("Error: Could not read the image.")
        return

    # Preprocess the original image
    preprocessed_image = preprocess_image(original_image)
    
    # Warp the preprocessed image
    warped_image = perspective_warp(preprocessed_image)
   
    # Detect arcs on the warped image
    circles = detect_arcs(warped_image, min_radius=50, max_radius=300)
    
    #Mean center and radius
    resulted_image = find_center(circles)

    # Create detailed heatmap from circles
    heatmap = create_detailed_heatmap(warped_image, circles)

    # Isolate the prominent arc
    prominent_arc = isolate_prominent_arc(heatmap)

    # Visualize results
    result_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
    
    # Draw all detected circles (optional, for visualization)
    for (x, y, r) in circles:
        cv2.circle(result_image, (x, y), r, (0, 255, 0), 1)

    # Draw the prominent arc
    if prominent_arc is not None:
        cv2.drawContours(result_image, [prominent_arc], 0, (0, 0, 255), 2)
        arc_length = cv2.arcLength(prominent_arc, False)
        print(f"Prominent Arc Length: {arc_length:.2f}")
    else:
        print("No prominent arc detected")

    # Display results
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Preprocessed Image", preprocessed_image)
    cv2.imshow("Warped Image", warped_image)
    cv2.imshow("Circle Heatmap", heatmap * 255)  # Multiply by 255 to make it visible
    cv2.imshow("Arc Detection Result", result_image)
    cv2.imshow("Mean Radius", resulted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
def sliding_window_top(warped_image, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(warped_image[:warped_image.shape[1]//2, :], axis=0)
    midpoint = np.int32(histogram.shape[1]//2)
    top = np.argmax(histogram[:midpoint])
    
    window_height = np.int32(warped_image.shape[0]//nwindows)
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    topx_current = topx_base
    top_lane_inds = []