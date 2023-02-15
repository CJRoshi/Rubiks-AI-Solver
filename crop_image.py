import copy
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np


#This function returns a canny image.
def detect_edges(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # apply Canny edge detection
    canny = cv2.Canny(blurred, 50, 150)
    
    return canny

# Sanity checks
def is_closed(contour):
    # Check if the first and last points of the contour are close enough
    if cv2.contourArea(contour) >= 400:
        print(np.allclose(contour[0], contour[-1], 20, 20))
        return np.allclose(contour[0], contour[-1],20, 20)
    else:
        return False

    

def is_in_region(contour, region):
    # Check if all points of the contour are inside the specified region
    return np.all([cv2.pointPolygonTest(region, tuple(point), False) >= 0 for point in contour])

'''----- MAIN -----'''

# init camera (1 refers to the rear cam)
cam = cv2.VideoCapture(1)

# allow shutter to open/autofocus
time.sleep(0.5)

result, image = cam.read()

while result:
    # Capture video frame by frame
    ret, frame = cam.read()

    # Display the resulting frame
    cv2.imshow('VIDEO FEED', frame)
      
    #Use Q to quit
    if cv2.waitKey(81)  == ord('q'):
        break

    #Use P to capture and process images
    if cv2.waitKey(80) == ord('p'):
        
        #Dimensions to crop center image of camera, adjusted dynamically by resolution
        border_top  = (len(frame))//8
        border_side = (len(frame[0]))//2-3*border_top
        center_width = 6*border_top

        # Crop image
        frame = frame[border_top:border_top+center_width, border_side:border_side+center_width]

        # Detect edges in the image
        edges = detect_edges(frame)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, contours, -1, (163, 100, 38), 3)

        #FOR EACH REGION
        width1 = len(edges)
        width2 = len(edges[0])
        n=3

        for hor in range(n):
            for ver in range(n):
                unit_width = width1//n
                unit_height = width2//n

                region = edges[hor*unit_width:(hor+1)*unit_width, ver*unit_height:(ver+1)*unit_height]

                # Iterate through the contours and hierarchy
                for i, contour in enumerate(contours):
                    # Check if the contour is outermost
                    if hierarchy[0][i][3] == -1:
                        # Check if the contour is closed
                        if is_closed(contour):
                            # Check if the contour is in one region of the image
                            if is_in_region(contour, region):
                                # Draw the contour on the image
                                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

            # Display the edges and cropped image.
            cv2.imshow('Edges', edges)
            cv2.imshow('Cropped', frame)



    
            






# After the loop, release the camera object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()
