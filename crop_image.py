import numpy as np
import cv2
import time
import copy

#This function returns a canny image.
def detect_edges(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # apply Canny edge detection
    canny = cv2.Canny(blurred, 50, 150)
    
    return canny

# For the purposes of finding regions....


def flood_Fill(image, x, y, color, visited):
    # Check if x, y are within the bounds of the image and the pixel is not already visited
    if x >= 0 and x < len(image) and y >= 0 and y < len(image[0]) and visited[x][y] == 0:
        # Mark the pixel as visited
        visited[x][y] = 1
        # Recursively color all the surrounding pixels
        flood_Fill(image, x + 1, y, color, visited)
        flood_Fill(image, x - 1, y, color, visited)
        flood_Fill(image, x, y + 1, color, visited)
        flood_Fill(image, x, y - 1, color, visited)

def count_Regions(image):
    visited = [[0 for j in range(len(image[0]))] for i in range(len(image))]
    regions = 0
    for i in range(len(image)):
        for j in range(len(image[0])):
            if visited[i][j] == 0:
                regions += 1
                flood_Fill(image, i, j, regions + 1, visited)
    return regions


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
        cv2.drawContours(frame, contours, -1, (163, 100, 38), 3)

        # Display the edges and cropped image.
        cv2.imshow('Edges', edges)
        cv2.imshow('Cropped', frame)


# After the loop, release the camera object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()
