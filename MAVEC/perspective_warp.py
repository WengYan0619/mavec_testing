import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, image = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    height, width = image.shape[:2]

    angle = 35

    box_height =  2/3*image.shape[0]

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


    # Convert images for display
    original_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    warped_display = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # Plot the original and transformed images
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_display)
    plt.title('Original Image with Source Points')
    plt.scatter(src[:, 0], src[:, 1], color='red') # Marking the points

    plt.subplot(1, 2, 2)
    plt.imshow(warped_display)
    plt.title('Warped Image')
    plt.show()


    # Display the resulting frame
    #cv2.imshow('Live Video Stream', warped_display)
    #cv2.imshow('Live Video Stream', original_display)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

