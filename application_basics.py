"""
APPLICATION BASICS
Author: Nino R, Blanca S, Alekhya P, Caden B.
Date: ** IN DEVELOPMENT **
This code implements a very basic form of the application flow, though without cube-solving functionality.
"""

### IMPORT ###
import time
import dropbox
import cv2
import numpy as np
import os
from PIL import Image
from image_segemtation import guidelines
from keras.models import load_model

### CONST ###

# Load AI model.
model = load_model('./model_checkpoint')

'''____________________________ FUNC __________________________________'''
import requests

def get_dbx_filenames(dbx) -> list:
    '''
    Get the filenames from a Dropbox, dbx.
    '''
    filenames = [str(entry.name) for entry in dbx.files_list_folder('').entries]
    return filenames
    

def get_next_available_num(filenames) -> int:
    '''
    Loop over a list of files in the form "edge_colstring_size_num.png" and find the next available num.
    '''
    used_numbers = set()
    for filename in filenames:
        parts = filename.split("_")
        if len(parts) == 4 and parts[-1].endswith(".png"):
            num = parts[-1].split(".")[0]
            used_numbers.add(num)

    next_num = 0
    while str(next_num) in used_numbers:
        next_num += 1

    return next_num

def upload_to_dropbox(img, size, edged, colstring, token):

    # Setub Dropbox for reception
    dbx = dropbox.Dropbox(token)
    filenames = get_dbx_filenames(dbx)
    next_num = get_next_available_num(filenames)

    # Preparing image file, filename.
    img = Image.fromarray(img)
    filename = edged+"_"+colstring+"_"+size+"_"+str(next_num)+".png"

    # Save the image as a PNG file.
    img.save(filename, "PNG")

    # Read the photo file and upload.
    with open(filename, "rb") as file:
        response = dbx.files_upload(file.read(), "/" + filename)
    
    # Remove the local file.
    os.remove(filename)

def analyze_photo(img:np.ndarray, model) -> tuple[str, str, str, np.ndarray]:
    '''
    Given an RGB image, preprocess the image and place it in a form for RubiksNet Predictions.

    Feed it to RubiksNet, and then process the results. 

    Return the results, and also return the processed version of the image.
    '''



    ### IMAGE PROCESSING ###
    imgheight, imgwidth, depth = np.shape(img)

    if imgwidth >= imgheight:
        larger_side = imgwidth
        smaller_side = imgheight

        border_top  = (smaller_side)//8
        border_side = (larger_side)//2-3*border_top
        center_width = 6*border_top

    elif imgwidth < imgheight:
        larger_side = imgheight
        smaller_side = imgwidth

        border_side  = (smaller_side)//8
        border_top = (larger_side)//2-3*border_side
        center_width = 6*border_side

    # Crop the img to the above dimensions.
    img_cropped = np.copy(img)
    img_cropped = img_cropped[border_top:border_top+center_width, border_side:border_side+center_width]
    
    # Shape image to model's expectations...
    img_cropped = Image.fromarray(img_cropped)
    img_cropped = img_cropped.resize((360,360))
    img_cropped = np.asarray(img_cropped)

    # Return this processed version.
    img_2 = img_cropped.copy()

    # Final image for AI.
    img_cropped = np.reshape(img_cropped, (1,360,360,3))


    ### AI PREDICTS ###
    predictions = model.predict(img_cropped)

    results = [tensor[0] for tensor in predictions]

    # Size branch results.
    if int(results[0][0])== 1:
        size = 'large'
        #print("AI predicts this is a close-up.")
    else:
        size = 'small'
        #print("AI predicts this is not a close-up.")

    # Edged branch results.
    if int(results[1][0]) == 1:
        edged = 'edge'
        #print("AI predicts this is an edged cube.")
    else:
        edged = 'noedge'
        #print("AI predicts this is an edgeless cube.")

    # Processing all 9 color branches and their results.
    colstring = ''
    for square in range(2,11):
        if results[square][0] == max(results[square]):
            colstring += 'R'
        elif results[square][1] == max(results[square]):
            colstring += 'O'
        elif results[square][2] == max(results[square]):
            colstring += 'Y'
        elif results[square][3] == max(results[square]):
            colstring += 'G'
        elif results[square][4] == max(results[square]):
            colstring += 'B'
        elif results[square][5] == max(results[square]):
            colstring += 'W'

    return size, edged, colstring, img_2

'''________________________________ MAIN _______________________________'''

def main():
    # Initialize the camera object.
    cam = cv2.VideoCapture(1)
    result, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Time delay to allow the shutter to autofocus.
    time.sleep(0.5)

    while result:
        # Capture video frame by frame
        result, img = cam.read()
        

        # Place guidelines on a copy of the img.
        img_guidelines = guidelines(img)
        
        # Display the resulting copy.

        cv2.imshow(('VIDEO FEED'), img_guidelines) 
        
        # Use Q to quit.
        if cv2.waitKey(81)  == ord('q'):
            break

        # Use P to capture and process images.
        if cv2.waitKey(80) == ord('p'):

            # Convert Camera BGR to Image RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Recieve Model Predictions
            size, edged, colstring, img = analyze_photo(img, model)
            print(colstring)

            # User Corrections
            colstring = input("Correct the colorstring.\nKEY:\nR -> Red | O -> Orange | Y -> Yellow | G -> Green | B -> Blue | W -> White\n")
            
            if edged == 'edge':
                print("The AI predicts this is an edged cube. Is it?")
                user_edgecorrect = input("Y or Yes for YES, N or No for NO").lower()
                while True:
                    if user_edgecorrect in ['y', 'yes']:
                        break
                    elif user_edgecorrect in ['n', 'no']:
                        edged = 'noedge'
                        break
                    else:
                        user_edgecorrect = input("Invalid input!\nY or Yes for YES, N or No for NO").lower()
            elif edged == 'noedge':
                print("The AI predicts this is an edgeless cube. Is it?")
                user_edgecorrect = input("Y or Yes for YES, N or No for NO").lower()
                while True:
                    if user_edgecorrect in ['y', 'yes']:
                        break
                    elif user_edgecorrect in ['n', 'no']:
                        edged = 'edge'
                        break
                    else:
                        user_edgecorrect = input("Invalid input!\nY or Yes for YES, N or No for NO").lower()

            if size == 'large':
                print("The AI predicts this is a close-up shot.\n(Meaning, you cannot see much of the background inside the guidelines.)\nIs it?")
                user_edgecorrect = input("Y or Yes for YES, N or No for NO").lower()
                while True:
                    if user_edgecorrect in ['y', 'yes']:
                        break
                    elif user_edgecorrect in ['n', 'no']:
                        size = 'small'
                        break
                    else:
                        user_edgecorrect = input("Invalid input!\nY or Yes for YES, N or No for NO").lower()
            elif edged == 'noedge':
                print("The AI predicts this isn't a close-up shot.\n(Meaning, you can see some background inside the guidelines.)\nIs it?")
                user_edgecorrect = input("Y or Yes for YES, N or No for NO").lower()
                while True:
                    if user_edgecorrect in ['y', 'yes']:
                        break
                    elif user_edgecorrect in ['n', 'no']:
                        size = 'large'
                        break
                    else:
                        user_edgecorrect = input("Invalid input!\nY or Yes for YES, N or No for NO").lower()

                


            # Upload to Dropbox
            upload_to_dropbox(img, size, edged, colstring, token='sl.BgX0YrzfOJn8ssGQ8v7-R8S4xJBNuLTp-YtUH8IILVwVVi-B9O6Hw9gYLLg3C-g1pp2lhB_j4phdeb1tpHjeMwMSbHtLIPW2FPHGnqp69d8YD4yGpZ2Y9jlvPatc-hjJaUWdyjcd')

            cv2.destroyAllWindows()

        # After the loop, release the camera object.
    cam.release()

        # Destroy all windows.
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()