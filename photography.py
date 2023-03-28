import time
import cv2
import numpy as np
from crop_image import guidelines

# Initialize the camera object.
cam = cv2.VideoCapture(1)
result, img = cam.read()

# Time delay to allow the shutter to autofocus.
time.sleep(0.5)
image_num=48
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
        cv2.imwrite('rubiks_img_'+str(image_num)+'.png', img)
        image_num+=1
        cv2.destroyAllWindows()
        time.sleep(0.75)

    # After the loop, release the camera object.
cam.release()

    # Destroy all windows.
cv2.destroyAllWindows()