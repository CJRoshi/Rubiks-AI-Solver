import time
import cv2
import numpy as np
from crop_image import guidelines

# Initialize the camera object.
cam = cv2.VideoCapture(1)
result, img = cam.read()

# Time delay to allow the shutter to autofocus.
time.sleep(0.5)
image_num=75

folder = "dataset_imgs/"

while result:
    # Capture video frame by frame
    result, img = cam.read()

    # Place guidelines on a copy of the img.
    img_guidelines = guidelines(img)

    # Display the resulting copy.
    cv2.imshow(('VIDEO FEED -- Captured '+str(image_num)+' images.'), img_guidelines)
        
    # Use Q to quit.
    if cv2.waitKey(81)  == ord('q'):
        break

    # Use P to capture and process images.
    if cv2.waitKey(80) == ord('p'):

        ### IMAGE PROCESSING ###

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
        
        ### FILE CLASSIFICATION ###
        edge = "edge_" if "y" in input("Is this an edged cube? (Y/N)\n").lower() else "noedge_"
        print("Write the colors on this face in sequence, such as 'WWWWWWWWW' for all-white or 'ROYGBWBGY'.")
        print("R is RED\nO is ORANGE\nY is YELLOW\nG is GREEN\nB is BLUE\nW is WHITE\n")
        colstring = input().upper() + "_"
        small = "large_" if "y" in input("Is this a closeup? (Y/N)\n").lower() else "small_"
        cv2.imwrite((folder+edge+colstring+small+str(image_num)+'.png'), img)
        image_num+=1
        
        cv2.destroyAllWindows()
        time.sleep(0.75)

    # After the loop, release the camera object.
cam.release()

    # Destroy all windows.
cv2.destroyAllWindows()