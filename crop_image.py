import numpy as np
import cv2
import time

def detect_edges(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # apply Canny edge detection
    canny = cv2.Canny(blurred, 50, 150)
    
    return canny

# init camera (1 refers to the rear cam)
cam = cv2.VideoCapture(1)

# allow shutter to open/autofocus
time.sleep(3)

result, image = cam.read()

# Show image in window
cv2.imshow("frame", image)


while result:
    # Capture the video frame
    # by frame
    ret, frame = cam.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(81)  == ord('q'):
        break

    if cv2.waitKey(13) == ord('\r'):
        print(frame)

        # Detect edges in the image
        edges = detect_edges(frame)

        # Display the edges
        cv2.imshow('Edges', edges)

        
  
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()
