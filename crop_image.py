### IMPORT ###
import time
import cv2
import numpy as np
import os
import pandas as pd

### CONSTANTS ###

# For the "Cube face format", neighbors defines which squares are neighbors of the others either horizontally or vertically.
# Neighbors_diag is similar, but for diagonal neighbors.


'''
For neighbors:
if square2 == square1 + 1:
    square2 is RIGHT of square1
elif square2 == square1 - 1:
    square2 is LEFT of square1
elif sqaure2 == square1 + 3:
    square2 is BELOW square1
elif square2 == square1 - 3:
    square2 is ABOVE square1

For neighbors_diag:
if square2 == square1 + 4:
    square2 is DOWN-LEFT of square1
elif square2 == square1 + 2:
    square2 is DOWN-RIGHT of square1
elif square2 == square1 - 4:
    square2 is UP_LEFT of square1
elif square2 == square1 - 2:
    square2 is UP_RIGHT of square1
'''

### FUNC ###
def get_center(contour:np.ndarray) -> tuple[int, int]:
    '''
    A function that gets the center of a contour.
    '''
    moments = cv2.moments(contour)
    contour_x = int(moments['m10']/moments['m00'])
    contour_y = int(moments['m01']/moments['m00'])

    return contour_x, contour_y

def side_length(contour:np.ndarray) -> int:
    '''
    A function that gets a side-length for a square with a roughly equal area to the contour.
    '''

    A = cv2.contourArea(contour)
    s = int(np.sqrt(A))

    return s

def dist_finder(square1:int, square2:int, label_key:dict[int:int], sticker_contours:list[np.ndarray]) -> int or None:
    '''
    A function that finds the distance between the centers of two squares on the Rubik's cube, based on sticker_contours.

    Inputs:
    
    square1 (int):              The number of the first square. Used to find its contour in sticker_contours.
    square2 (int):              The number of the second square. Used to find its contour in sticker_contours.
    label_key (dict[int:int]):           A dict where each square's number is matched with its index in sticker_contours.
    sticker_contours (list[np.ndarray]):    A list of the contours of possible stickers on the Rubik's Cube.

    Output:
    dist (int): The distance between the centers of two directly adjacent squares on a Rubik's cube. 
    None returned if no relation exists between square1 and square2.
    '''

    center1x, center1y = get_center(sticker_contours[label_key[square1]])
    center2x, center2y = get_center(sticker_contours[label_key[square2]])

    if square2 == square1 + 1 or square2 == square1 - 1 or square2 == square1 + 3 or square2 == square1 - 3:
        dist = int(np.sqrt(((abs(center2x-center1x))**2+(abs(center2y-center1y))**2)))
    elif square2 == square1 + 4 or square2 == square1 + 2 or square2 == square1 - 4 or square2 == square1 - 2:
        dist = int(np.sqrt(((abs(center2x-center1x))**2+(abs(center2y-center1y))**2))/np.sqrt(2))
    else:
        return None
    return dist
       
def avg_dist_from_square(square:int, label_key:dict[int:int], sticker_contours:list[np.ndarray]) -> int or None:
    '''
    A function that averages all the dists created by dist_finder for a given square.
    '''
    existing_squares = []
    for squarenum, contournum in label_key.items():
        existing_squares.append(squarenum)

    
    if existing_squares != None:
        potential_dists = []
        if square in existing_squares:
            existing_squares.remove(square)
            for remaining_square in existing_squares:
                if dist_finder(square, remaining_square, label_key, sticker_contours) != None:
                    potential_dists.append(dist_finder(square, remaining_square, label_key, sticker_contours))
    else:
        return None
    return int(np.average(potential_dists))

def average_dist(label_key:dict[int:int], sticker_contours:list[np.ndarray]) -> int or None:
    '''A function that averages all the dists created by avg_dist_from_square for a given face on the Rubik's Cube.'''
    dists = []
    for squarenum, contournum in label_key.items():
        dist_from_square = avg_dist_from_square(squarenum, label_key, sticker_contours)
        if dist_from_square != None:
            dists.append(dist_from_square)
    if dists != None:
        return int(np.average(dists))
    else:
        return None

def create_mask(img:np.ndarray, contour:np.ndarray) -> np.ndarray:
    '''
    A function that generates an image mask given an image and the contour within.
    '''
    mask = np.zeros_like(img)
    cv2.drawContours(image=mask, contours=[contour], contourIdx=-1, color=[255,255,255], thickness=-1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask

def statistics(img:np.ndarray, mask:np.ndarray) -> tuple[list[(str, cv2.calcHist)], list[int,int,int], float]:

    '''
    A function that grabs some useful statistics from a region of an image using an image and a mask made by mask().
    '''
    # Generating color histograms
    histogram_list = []
    color = ('b','g','r')
    for i,col in enumerate(color):
        histogram_list.append((col,cv2.calcHist([img],[i],mask,[256],[0,256])))

    # Mean and standard devation of color        
    mean, stdev = cv2.meanStdDev(src=img, mask=mask)

    return (histogram_list, np.ravel(mean), np.ravel(stdev))

def generate_square_contour(x:int, y:int, s:int) -> list[tuple[int,int]]:
    '''
    A function that generates a contour of a square.
    '''
    # Generate x coordinates for the top and bottom edges of the square
    x_top = np.linspace(x, x+s-1, num=s)
    x_bottom = np.linspace(x, x+s-1, num=s)

    # Generate y coordinates for the left and right edges of the square
    y_left = np.linspace(y+1, y+s-2, num=s-2)
    y_right = np.linspace(y+1, y+s-2, num=s-2)

    # Combine the coordinates for each edge into a single array
    x_coords = np.concatenate([x_top, np.full(s-2, x+s-1), np.flip(x_bottom), np.full(s-2, x)])
    y_coords = np.concatenate([np.full(s, y), y_left, np.full(s, y+s-1), np.flip(y_right)])

    
    # Reshape the coordinates into the format expected by cv2.drawContours()
    contour = np.column_stack((x_coords, y_coords)).reshape((-1, 1, 2)).astype(np.int32)

    return contour

def geometry_fill_in(label_key:dict[int:int], sticker_contours:list[np.ndarray], img:np.ndarray) -> tuple[dict[int:int], list[np.ndarray]]:
    '''
    A function that takes an an image of one face of a Rubik's cube with some stickers identified, and attempts to identify the remaining stickers using basic geometry.

    Inputs:
    label_key (dict):           A dict where each square's number is matched with its index in sticker_contours.
    sticker_contours (list):    A list of the contours of possible stickers on the Rubik's Cube.
    img (np.ndarray):           An image of one face of the Rubik's Cube, cropped by parent function identify_stickers.

    Outputs:
    label_key (dict):           A dict where each square's number is matched with its index in sticker_contours.
    sticker_contours (list):    A list of the contours of possible stickers on the Rubik's Cube; this should have more stickers now.

    This function first checks through what's already been found and computes a distance between two adjacent squares for the cube.

    Then, it checks which squares have not yet been found, and it "hallucinates" one.

    If the region over this fake square has a low standard deviation, it's solid-colored and therefore a decent candidate for a sticker.
    We add this contour to sticker_contours and update label_key accordingly.
    '''
    neighbors = {0:[1,3], 1:[0,2,4], 2:[1,5], 3:[0,4,6], 4:[1,3,5,7], 5:[2,4,8], 6:[3,7], 7:[4,6,8], 8:[5,7]}
    neighbors_diag = {0:[4], 1:[3, 5], 2:[4], 3:[1,7], 4:[0,2,6,8], 5:[1,7], 6:[4], 7:[3, 5], 8:[4]}

    avg_dist = average_dist(label_key, sticker_contours)

    label_key2 = label_key.copy()
    
    for square, contournum in label_key.items():
        contour = sticker_contours[contournum]

        candidates = neighbors[square]+neighbors_diag[square]
        for neighbor in candidates:
            x, y = get_center(contour)
            if neighbor not in label_key.keys():

                # Hallucinate a square!

                # Adjust center coords to hallucinate properly.
                if square == neighbor + 1:
                    x+=avg_dist
                elif square == neighbor - 1:
                    x-=avg_dist
                elif square == neighbor + 3:
                    y+=avg_dist
                elif square == neighbor - 3:
                    y-=avg_dist
                elif square == neighbor + 4:
                    x+=avg_dist
                    y+=avg_dist
                elif square == neighbor + 2:
                    x-=avg_dist
                    y+=avg_dist
                elif square == neighbor - 4:
                    x+=avg_dist
                    y-=avg_dist
                elif square == neighbor - 2:
                    x-=avg_dist
                    y-=avg_dist
                else: 
                    continue

                s = side_length(contour)
                
                x-=s//2
                y-=s//2

                square_contour = generate_square_contour(x,y,s)

                mask = create_mask(img, square_contour)
                histogram_list, mean, stdev = statistics(img, mask)

                if np.average(stdev) <= 20:
                    sticker_contours.append(square_contour)
                    label_key2[neighbor] = len(sticker_contours)-1
        else:
            continue

    return label_key2, sticker_contours

def guidelines(img:np.array) -> np.array:
    '''
    A function that applies guidelines to a raw image.

    Input:
    img (np.ndarray):               A raw RGB image of a face of a Rubik's Cube.

    Output:
    img_guidelines (np.ndarray):    A copy of img with guidelines over its center portion.

    Guidelines are placed using a cropping algorithm identical to identify_stickers.
    '''
    # Get the dimensions to crop the img to its center plus some bound of error, adjusted dynamically by resolution.
    imgheight, imgwidth, depth = np.shape(img)
    
    img_guidelines = np.copy(img)
    WHITE = [255, 255, 255]

    if imgwidth >= imgheight:
        larger_side = imgwidth
        smaller_side = imgheight

        border_top  = (smaller_side)//8
        border_side = (larger_side)//2-3*border_top
        center_width = 6*border_top

        # Horizontal borders.

        # Top
        img_guidelines[border_top][border_side:border_side+center_width] = WHITE

        # Thirds
        img_guidelines[border_top+center_width//3][border_side:border_side+center_width] = WHITE
        img_guidelines[border_top+2*center_width//3][border_side:border_side+center_width] = WHITE

        # Bottom 
        img_guidelines[border_top+center_width][border_side:border_side+center_width] = WHITE

        # Vertical borders.

        # Left
        img_guidelines[border_top:border_top+center_width,border_side] = WHITE

        # Thirds
        img_guidelines[border_top:border_top+center_width,border_side+center_width//3] = WHITE
        img_guidelines[border_top:border_top+center_width,border_side+2*center_width//3] = WHITE

        # Right
        img_guidelines[border_top:border_top+center_width,border_side+center_width] = WHITE

    elif imgwidth < imgheight:
        larger_side = imgheight
        smaller_side = imgwidth

        border_side  = (smaller_side)//8
        border_top = (larger_side)//2-3*border_side
        center_width = 6*border_side

        # Horizontal borders.

        # Top
        img_guidelines[border_top][border_side:border_side+center_width] = WHITE

        # Thirds
        img_guidelines[border_top+center_width//3][border_side:border_side+center_width] = WHITE
        img_guidelines[border_top+2*center_width//3][border_side:border_side+center_width] = WHITE

        # Bottom 
        img_guidelines[border_top+center_width][border_side:border_side+center_width] = WHITE

        # Vertical borders.

        # Left
        img_guidelines[border_top:border_top+center_width,border_side] = WHITE

        # Thirds
        img_guidelines[border_top:border_top+center_width,border_side+center_width//3] = WHITE
        img_guidelines[border_top:border_top+center_width,border_side+2*center_width//3] = WHITE

        # Right
        img_guidelines[border_top:border_top+center_width,border_side+center_width] = WHITE

    return img_guidelines

def detect_edges(img:np.ndarray, lower:int=0, upper:int=100) -> np.ndarray:
    '''
    A function that applies Canny Edge Detection to an image.

    Input:
    img (np.ndarray):           An RGB image.
    lower (int, default 50):    OPTIONAL, The lower threshold for cv2.Canny.
    upper (int, default 150):   OPTIONAL, The upper threshold for cv2.Canny.


    Output:
    canny (np.ndarray):   A 1-bit color array of the edges in img.

    img is converted to grayscale, and Gaussian Blur is applied in order to reduce noise.
    Canny edge detection is then applied to the denoised image.
    '''
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # apply Canny edge detection
    canny = cv2.Canny(gray, lower, upper)
    #cv2.imshow("Canny Edges", canny)
    
    return canny

def detect_edges_iterative(img:np.array, mask:np.array) -> tuple[np.ndarray, list[np.ndarray]]:
    '''
    A function that finds edges within a masked region.
    '''
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    edges = detect_edges(img_masked)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return img_masked, contours

def is_rubik_square(contour:np.ndarray, center_width:int) -> bool:
    '''
    A function that looks at a given contour and determines whether or not it is the acceptable size and type for a Rubik's Cube sticker.

    Inputs:
    contour (np.ndarray):     An array of [x,y] points in some img that define the boundary of a region.
    center_width:           The INTENDED width of the square center region, in pixels. This value is determined from the height of the original raw photo.

    Output:
    bool:                   Evaluates to True if contour has an accaptable area and is closed; otherwise, False.
    '''
    # If the area of contour is within a given margin of error (+/- 1/3 of the expected length)...
    if cv2.contourArea(contour) >= (center_width/3-center_width/9)**2 and cv2.contourArea(contour) <= (center_width/3+center_width/9)**2:
        # Return True if the contour is closed; otherwise, return False.
        return np.allclose(contour[0], contour[-1], 30, 30)
    else:
        return False
    
def identify_stickers(img:np.ndarray, lower:int=0, upper:int=100) -> tuple[np.ndarray, list]:
    '''
    A function that identifies the stickers on one face of a Rubik's Cube.

    Input:
    img (np.ndarray):           An RGB image that is a raw photo of one face of the Rubik's Cube. 

    Outputs:
    img_cropped (np.ndarray):   img, after dynamic cropping and basic adjustments have been applied to it.
    sticker_contours (list):    A list of the contours; these contours are the boundaries of the stckers.
    lower (int, default 50):    OPTIONAL, The lower threshold for cv2.Canny in detect_edges.
    upper (int, default 150):   OPTIONAL, The upper threshold for cv2.Canny in detect_edges.

    img is dynamically cropped to a center square plus one-eighth of the length of the largest side.
    Canny edge detection is applied using detect_edges, and cv2.findContours finds the contours that could correspond to stickers.

    The contours are processed using is_rubik_square, and then they are sorted by size. 
    The nine largest (which will be the Rubik's cube stickers) are returned, along with the cropped version of img.
    '''

    # Get the dimensions to crop the img to its center plus some bound of error, adjusted dynamically by resolution.
    imgheight, imgwidth, depth = np.shape(img)

    if imgwidth >= imgheight:
        larger_side = imgwidth
        smaller_side = imgheight

        border_top  = (smaller_side)//8
        border_side = (larger_side)//2-3*border_top
        true_center_width = 4*border_top
        center_width = 6*border_top

    elif imgwidth < imgheight:
        larger_side = imgheight
        smaller_side = imgwidth

        border_side  = (smaller_side)//8
        border_top = (larger_side)//2-3*border_side
        true_center_width = 4*border_side
        center_width = 6*border_side

    # Crop the img to the above dimensions.
    img_cropped = np.copy(img)
    img_cropped = img_cropped[border_top:border_top+center_width, border_side:border_side+center_width]

    # Detect edges in the img.
    edges = detect_edges(img_cropped, lower, upper)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find all valid contours; sort them by size.
    closed_contours = []
    
    for contour in contours:
        if is_rubik_square(contour, true_center_width):
            closed_contours.append(contour)
    
    # Sort the valid contours by size.
    sorted_contours = sorted(closed_contours, key=cv2.contourArea, reverse=True)

    # Get the largest nine contours.
    sticker_contours = sorted_contours[:9]

    return img_cropped, sticker_contours

def correct_labels(img_cropped:np.ndarray, sticker_contours:list, display:bool=False) -> tuple[dict[int,int], list[np.ndarray]]:
    '''
    A function that takes cropped of a face of a Rubik's Cube and the locations of its stickers generated by identify_stickers, 
    which labels the stickers to a standard format:

    [0][1][2]
    [3][4][5]
    [6][7][8]

    In this format, 4 is the center, 0 is the top-left corner, etc.

    Inputs:
    img_cropped (np.ndarray):               The RGB image of the Rubik's Cube. It must be cropped to a square by identify_stickers in order for this algorithm to function.
    sticker_contours (list[np.ndarray]):    A list of the contours that are the stickers of the Rubik's Cube, also generated by identify_stickers.
    display (bool, default False):          OPTIONAL, Determines whether or not the results of this function are displayed in a cv2.imshow window.

    Output:
    label_key (dict[int:int]):              A dict where each square's number is matched with its index in sticker_contours.
    sticker_contours (list[np.ndarray]):    A list of the contours of possible stickers on the Rubik's Cube; input with some new squares added.
    '''
    # Preparation and coordinate generation for point tests. 
    point_tests = []
    label_key = {}
    standard_length = len(img_cropped)
    for y_coord_ind in range(3):
        for x_coord_ind in range(3):
            x_coord = standard_length//4*(x_coord_ind+1)
            y_coord = standard_length//4*(y_coord_ind+1)                
            point_tests.append((x_coord,y_coord))
    
    # Test each point with each contour; one should fit for each.
    for (cntindex, contour) in enumerate(sticker_contours):
        for (ptindex, point) in enumerate(point_tests):
            if cv2.pointPolygonTest(contour, point, False) > 0:
                label_key[ptindex]= cntindex

    '''
    # Try using geometry to fill in empty spaces.             
    if 2 < len(sticker_contours) < 9:
        label_key, sticker_contours = geometry_fill_in(label_key, sticker_contours, img_cropped)

    # Relabel
    for (cntindex, contour) in enumerate(sticker_contours):
        for (ptindex, point) in enumerate(point_tests):
            if cv2.pointPolygonTest(contour, point, False) > 0:
                label_key[ptindex]= cntindex
    '''

    if display:
        # Display the contours on the image, if specified.
        img_cropped_overwrite=np.copy(img_cropped)
        for ptindex, cntindex in label_key.items():
            contour_x, contour_y = get_center(sticker_contours[cntindex])
            cv2.putText(img_cropped_overwrite, text=str(ptindex+1), org=(contour_x, contour_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.drawContours(img_cropped_overwrite, list(sticker_contours[cntindex]), -1, color=(33,237,255))
            cv2.imshow('Stickers', img_cropped_overwrite)


    return label_key, sticker_contours

######## MAIN ########
if __name__=="__main__":
    # Initialize the camera object.
    cam = cv2.VideoCapture(1)
    result, image = cam.read()
    # Time delay to allow the shutter to autofocus.
    time.sleep(0.5)
    while result:
        # Capture video frame by frame
        result, image = cam.read()
        # Place guidelines on a copy of the image.
        image_guidelines = guidelines(image)
        # Display the resulting copy.
        cv2.imshow('VIDEO FEED', image_guidelines)
        
        # Use Q to quit.
        if cv2.waitKey(81)  == ord('q'):
            break
        # Use P to capture and process images.
        if cv2.waitKey(80) == ord('p'):
            image_cropped, sticker_contours = identify_stickers(image)
            label_key = correct_labels(image_cropped, sticker_contours, True)
    # After the loop, release the camera object.
    cam.release()
    # Destroy all windows.
    cv2.destroyAllWindows()