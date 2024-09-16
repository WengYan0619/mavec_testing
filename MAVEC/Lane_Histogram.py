import cv2
import numpy as np
import matplotlib.pyplot as plt

def ROI(image):
    # Define points for the quadrilateral
    points = np.array([[(322, 212), (100, 470), (560, 470), (451, 212)]], dtype=np.int32)
    # Create a mask same size as the image, but with all zeros
    mask = np.zeros_like(image)
    # Fill the defined polygon with white (255)
    cv2.fillPoly(mask, points, 255)
    # Apply the mask to the image using bitwise and
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



# Load the image
image = cv2.imread(r"C:\Users\wengy\Downloads\image3.png")

# Apply the ROI to the original image (before or after converting to grayscale based on need)
roi_image = ROI(image)  # If ROI should be applied to the RGB image

gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)  # Convert ROI image to grayscale
# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply binary thresholding
_, binary = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)

def perspective_transformation(roi_image):
    src_points = np.float32([
        (322, 212),  # Top-left corner
        (100, 470),  # Bottom-left corner
        (560, 470),  # Bottom-right corner
        (451, 212)   # Top-right corner
    ])

    # Define destination points for a rectangle
    dst_points = np.float32([
        [200, 0],                        # New top-left corner, somewhat arbitrary
        [200, image.shape[0]],           # New bottom-left corner
        [image.shape[1]-200, image.shape[0]],  # New bottom-right corner
        [image.shape[1]-200, 0]          # New top-right corner
    ])

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(roi_image, M, (image.shape[1], image.shape[0]))
    
    return warped_image

# Perspective transformation function already defined
warped_image = perspective_transformation(roi_image)

# def draw_points(image, points, color=(0, 255, 0)):
#     img = image.copy()
#     for point in points:
#         cv2.circle(img, tuple(int(x) for x in point), 5, color, -1)
#     return img

# # Example usage
# image_with_src = draw_points(image, src_points, color=(255, 0, 0))  # Red for source points
# image_with_dst = draw_points(image, dst_points, color=(0, 255, 0))  # Green for destination points

# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(image_with_src, cv2.COLOR_BGR2RGB))
# plt.title('Source Points')
# plt.subplot(122)
# plt.imshow(cv2.cvtColor(image_with_dst, cv2.COLOR_BGR2RGB))
# plt.title('Destination Points')
# plt.show()


plt.subplot(1, 3, 3)
plt.imshow(warped_image )  # Perspective transformed image
plt.title('Warped Image')
plt.show()

