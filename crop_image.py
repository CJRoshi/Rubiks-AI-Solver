import numpy as np
import cv2
import time

# init camera (1 refers to the rear cam)
cam = cv2.VideoCapture(1)

# allow shutter to open/autofocus
time.sleep(1)

result, image = cam.read()

print(image)
if result:
  
    # Show image in window
    cv2.imshow("CALIBR8", image)
  
    # Save locally
    cv2.imwrite("CALIBR8.png", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
    cv2.waitKey(0)
    cv2.destroyWindow("CALIBR8")
  
# If captured image is corrupted/no camera exists
else:
    print("No image detected. Please try again.")