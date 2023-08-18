"""
COMMAND LINE APP
Author: Nino R, Blanca S, Alekhya P, Caden B.
Date: 7/19/2023
This code implements a very basic form of the application flow through the commmand line.
"""
### IMPORT ###
import json
import os
import subprocess
import time
import tkinter as tk
import webbrowser
import cv2
import dropbox
import numpy as np
from dropbox import DropboxOAuth2FlowNoRedirect
from keras.models import load_model
from PIL import Image

from cube_solving import cubesolutions
from image_segmentation import guidelines

# Load AI model as a constant.
# model = load_model('./model_checkpoint')

### FUNC ###

def analyze_photo(img: np.ndarray, model) -> tuple[str, str, str, np.ndarray]:
    '''
    Inputs:

    img (np.ndarray):       RGB image to be analyzed. It should be in the raw format from the camera without any cropping.

    model:                  A model preloaded by load_model; RubiksNet.

    
    Outputs:

    size (str):             Either "large" or "small". A judgement made by RubiksNet of how much space the Rubik's Cube takes up in img.
    
    edge (str):             Either "edge" or "noedge". A judgement by RubiksNet of whether or not the cube is an edgeless cube.

    colorstring (str):      A 9 letter string with the colors of one face of the Rubik's Cube, such as "WWWWWWWWW". 
                            Matches this format:

                                [0][1][2]

                                [3][4][5]

                                [6][7][8]

                            In this format, 4 is the center, 0 is the top-left corner, etc.
    
    prcsd_img (np.ndarray): An RGB image that is in the format expected by RubiksNet (360x360). Can be put into the dataset for training.
    

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

def capture_cube_face(center_color: str, face_name: str, model) -> tuple[str, str, str, np.ndarray]:
    '''
    Inputs:

    center_color (str): The current center color being requested, e.g. "Y". Used for instruction and to help guide the correction process.

    face_name (str):    The full name of the center color, e.g. "yellow".

    
    Outputs:

    size (str):             Either "large" or "small". A correction of RubiksNet's judgement in analyze_photo().
    
    edge (str):             Either "edge" or "noedge". A correction of RubiksNet's judgement in analyze_photo.

    colorstring (str):      A 9 letter string with the colors of one face of the Rubik's Cube, such as "WWWWWWWWW".
                            A correction of RubiksNet's judgement in analyze_photo. 
                            Matches this format:

                                [0][1][2]

                                [3][4][5]

                                [6][7][8]

                            In this format, 4 is the center, 0 is the top-left corner, etc.
    
    prcsd_img (np.ndarray): An RGB image that is in the format expected by RubiksNet (360x360). Can be put into the dataset for training.
    
    This function takes the user through the whole image capture and correction process. 
    It instructs the user to take photos and then helps correct RubiksNet's predictions, returning the corrections.
    '''
    # Instructions for photo capture
    if face_name == 'white':
        print(f"Position the cube so that the white center is forward and the red center is down.")
    elif face_name == 'red':
        print("Rotate up to red.")
    elif face_name == 'blue':
        print("Rotate left to blue.")
    elif face_name == 'orange':
        print("Rotate left to orange.")
    elif face_name == 'green':
        print("Rotate left again to green.")
    elif face_name == 'yellow':
        print("Rotate up one last time to yellow.")
    print("Press 'P' to capture the image of the cube face.")

    # Initialize the camera object.
    cam = cv2.VideoCapture(1)
    result, img = cam.read()
    time.sleep(0.5)

    # Image capture loop
    while result:
        # Capture video frame by frame
        result, img = cam.read()  # Create a copy of the captured image as disp_img.
        disp_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Place guidelines on the copied image
        disp_img = guidelines(disp_img)

        # Display the resulting image with guidelines
        cv2.imshow('VIDEO FEED', disp_img)

        # Use P to capture and process the image
        key = cv2.waitKey(80)
        if key == ord('p'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

    # Process and analyze the captured image
    size, edged, colstring, prcsd_img = analyze_photo(img, model)

    print("\nRubiksNet predicts the following colorstring:")
    print(colstring)

    # User Corrections

    # Colorstring corrections.
    colcorrect = input("\nCorrect the colorstring, if needed. (Left empty, nothing will be changed.)\nKEY:\nR -> Red | O -> Orange | Y -> Yellow | G -> Green | B -> Blue | W -> White\n")
    while True:
        # Successful conditions
        if colcorrect == '':
            break
        elif len(colcorrect) == 9 and colcorrect[4].upper() == center_color[0].upper():
            colstring = colcorrect.upper()
            break
        # Failing conditions
        elif len(colcorrect) != 9 and colcorrect[4].upper() == center_color[0].upper():
            print("Wrong length! You may have forgotten the last character or so.")
            colcorrect = input("Re-enter the colorstring.\n")
        elif (len(colcorrect) == 9 and colcorrect[4].upper() != center_color[0].upper()) or (len(colcorrect) != 9 and colcorrect[4].upper() != center_color[0].upper()):
            print("Wrong center character and/or wrong length. Retake the photo before correcting.")

            # Initialize the camera
            cam = cv2.VideoCapture(1)
            result, img = cam.read()
            time.sleep(0.5)

            # Image capture loop
            while result:
                # Capture video frame by frame
                result, img = cam.read()  # Create a copy of the captured image as disp_img.
                disp_img = img.copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Place guidelines on the copied image
                disp_img = guidelines(disp_img)

                # Display the resulting image with guidelines
                cv2.imshow('VIDEO FEED', disp_img)

                # Use P to capture and process the image
                key = cv2.waitKey(80)
                if key == ord('p'):
                    break
            
            cam.release()
            cv2.destroyAllWindows()

            # Process and reanalyze the new image.
            size, edged, colstring, prcsd_img = analyze_photo(img, model)
            colcorrect = input("Re-enter the colorstring.\n")
    
    # Edge corrections
    if edged == 'edge':
        print("\nThe AI predicts this is an edged cube. Is it?")
        user_edgecorrect = input("Y or Yes for YES, N or No for NO\n").lower()
        while True:
            if user_edgecorrect in ['y', 'yes']:
                break
            elif user_edgecorrect in ['n', 'no']:
                edged = 'noedge'
                break
            else:
                user_edgecorrect = input("Invalid input!\nY or Yes for YES, N or No for NO\n").lower()
    elif edged == 'noedge':
        print("\nThe AI predicts this is an edgeless cube. Is it?")
        user_edgecorrect = input("Y or Yes for YES, N or No for NO\n").lower()
        while True:
            if user_edgecorrect in ['y', 'yes']:
                break
            elif user_edgecorrect in ['n', 'no']:
                edged = 'edge'
                break
            else:
                user_edgecorrect = input("Invalid input!\nY or Yes for YES, N or No for NO\n").lower()

    # Size corrections
    if size == 'large':
        print("\nThe AI predicts this is a close-up shot.\n(Meaning, you cannot see much of the background inside the guidelines.)\nIs it?")
        user_edgecorrect = input("Y or Yes for YES, N or No for NO\n").lower()
        while True:
            if user_edgecorrect in ['y', 'yes']:
                break
            elif user_edgecorrect in ['n', 'no']:
                size = 'small'
                break
            else:
                user_edgecorrect = input("Invalid input!\nY or Yes for YES, N or No for NO\n").lower()
    elif size == 'small':
        print("\nThe AI predicts this isn't a close-up shot.\n(Meaning, you can see some background inside the guidelines.)\nIs it?")
        user_edgecorrect = input("Y or Yes for YES, N or No for NO\n").lower()
        while True:
            if user_edgecorrect in ['y', 'yes']:
                break
            elif user_edgecorrect in ['n', 'no']:
                size = 'large'
                break
            else:
                user_edgecorrect = input("Invalid input!\nY or Yes for YES, N or No for NO\n").lower()

    return size, edged, colstring, prcsd_img

def validate_checkerstring(checkerstring: str) -> bool:
    '''
    Inputs:

    checkerstring (str):    A collection of six colorstrings concatenated; a 54-long string representing the Rubik's Cube.

    Outputs:

    bool:                   True if this is a valid cube; False otherwise.

    A method that checks all cublets on the Cube to make sure this cube isn't impossible; 
    confirms that there are 9 cubelets of each color and that all characters are correct. 
    '''

    valid_chars = {'W', 'R', 'B', 'O', 'G', 'Y'}

    # Check for invalid characters
    invalid_chars = set(checkerstring) - valid_chars
    if invalid_chars:
        print("Invalid characters found:", invalid_chars)
        return False

    # Check if any color occurs in a count other than 9
    color_counts = {color: checkerstring.count(color) for color in valid_chars}
    invalid_counts = {color: count for color, count in color_counts.items() if count != 9}
    if invalid_counts:
        print("Invalid color counts:")
        for color, count in invalid_counts.items():
            print(f"{color}: {count}")
        return False

    return True

# Dropbox Uploading + dependencies.

TOKEN_FILE = "dropbox_token.json"

def generate_dropbox_token(app_key: str, app_secret: str) -> str:
    '''
    Inputs:

    app_key, app_secret (str):  The app_key and app_secret of RubiksNet's dataset on Dropbox. Used to do the authflow if needed.


    Outputs:

    token (str):                The OAuth2 Token used to access the dataset.

    A method that either loads an existing auth_code/token or has one be created by using Dropbox's API.

    If an OAuth2 Flow starts, it will open in browser, with a small tkinter window to enter the code in.
    '''

    # Attempt to load the token that already exists first, but if that doesn't work (code is invalid or does not exist), 
    # then go to the authflow.
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as token_file:
            token_data = json.load(token_file)
            token = token_data.get('access_token')
            if token:
                try:
                    dbx = dropbox.Dropbox(token)
                    dbx.users_get_current_account()
                    return token
                except dropbox.exceptions.AuthError:
                    pass
    
    # Start authorization flow
    auth_flow = DropboxOAuth2FlowNoRedirect(app_key, app_secret)
    authorize_url = auth_flow.start()

    webbrowser.open(authorize_url)  # Open the authorization URL in a web browser

    # Create a tk window to enter the auth code
    auth_window = tk.Toplevel()
    auth_window.title("RubiksNet Authorization")
    auth_code = tk.StringVar()
    label = tk.Label(auth_window, text="Please enter the authorization code:")
    label.pack(pady=10)
    entry = tk.Entry(auth_window, textvariable=auth_code)
    entry.pack(pady=5)

    result = {}  # Create an empty dictionary to hold the result

    def submit_code():
        '''Submit button behavior: stores auth code and destroys the GUI window.'''
        result['auth_code'] = auth_code.get()  # Store the auth code in the dictionary
        auth_window.destroy()

    # Submit button, GUI loop
    submit_button = tk.Button(auth_window, text="Submit", command=submit_code)
    submit_button.pack(pady=10)
    auth_window.mainloop()

    auth_code = result['auth_code']

    result = auth_flow.finish(auth_code)
    access_token = result.access_token

    with open(TOKEN_FILE, 'w') as token_file:
        json.dump({'access_token': access_token}, token_file)

    return access_token

def get_dbx_filenames(dbx) -> list:
    '''
    Get the filenames from a Dropbox, dbx.
    '''
    filenames = [str(entry.name) for entry in dbx.files_list_folder('').entries]
    return filenames
    
def get_next_available_num(filenames) -> int:
    '''
    Loop over a list of files in the form "edge_colstring_size_num.png" as they are in RubiksNet's Dataset 
    and find the next available num.
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

def upload_to_dropbox(img, size, edged, colstring):
    '''
    Inputs:

    img (np.ndarray): An RGB image that is in the format expected by RubiksNet (360x360). Can be put into the dataset for training.
    
    size (str):             Either "large" or "small". A judgement made by RubiksNet of how much space the Rubik's Cube takes up in img.
    
    edge (str):             Either "edge" or "noedge". A judgement by RubiksNet of whether or not the cube is an edgeless cube.

    colorstring (str):      A 9 letter string with the colors of one face of the Rubik's Cube, such as "WWWWWWWWW". 
                            Matches this format:

                                [0][1][2]

                                [3][4][5]

                                [6][7][8]

                            In this format, 4 is the center, 0 is the top-left corner, etc.
    
    This function takes a processed image and its metadata and uploads it to RubiksNet's Dataset on Dropbox.
    '''
    # Get token
    app_key = '4qcw82f7y6sa9do'
    app_secret = '7jc0gnpah527k4p'
    token = generate_dropbox_token(app_key=app_key, app_secret=app_secret)

    # Setup Dropbox for reception
    dbx = dropbox.Dropbox(token)
    filenames = get_dbx_filenames(dbx)
    next_num = get_next_available_num(filenames)

    # Preparing image file, filename
    img = Image.fromarray(img)
    filename = edged+"_"+colstring+"_"+size+"_"+str(next_num)+".png"

    # Save the image as a PNG file
    img.save(filename, "PNG")

    # Read the photo file and upload
    with open(filename, "rb") as file:
        response = dbx.files_upload(file.read(), "/" + filename)

    # Remove the local file
    os.remove(filename)


def download_dataset_from_dropbox():
    '''
    Helper function. Downloads all of RubiksNet's dataset.
    '''

    # Get token
    app_key = '4qcw82f7y6sa9do'
    app_secret = '7jc0gnpah527k4p'
    token = generate_dropbox_token(app_key=app_key, app_secret=app_secret)

    # Access Dropbox
    dbx = dropbox.Dropbox(token)

    # Define the Dropbox path of the dataset folder
    dropbox_dataset_folder = ''

    # Define the local folder where the dataset will be downloaded
    local_dataset_folder = './dataset_imgs'

    # Create the local folder if it doesn't exist
    os.makedirs(local_dataset_folder, exist_ok=True)

    # Get a list of files in the Dropbox dataset folder
    result = dbx.files_list_folder(dropbox_dataset_folder)

    # Download each file from Dropbox
    for entry in result.entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            # Get the file name and full path on Dropbox
            file_name = entry.name
            dropbox_file_path = entry.path_display

            # Define the local file path where the file will be downloaded
            local_file_path = os.path.join(local_dataset_folder, file_name)

            # Download the file from Dropbox
            dbx.files_download_to_file(local_file_path, dropbox_file_path)

    print("Dataset downloaded.")

def update_dataset_from_dropbox():
    '''
    Helper function. Updates the existing local dataset with any new files in RubiksNet's dataset.
    '''

    # Get token
    app_key = '4qcw82f7y6sa9do'
    app_secret = '7jc0gnpah527k4p'
    token = generate_dropbox_token(app_key=app_key, app_secret=app_secret)

    # Access Dropbox
    dbx = dropbox.Dropbox(token)

    # Define the Dropbox and local dataset folders
    dropbox_dataset_folder = ''
    local_dataset_folder = './dataset_imgs'

    # Get a list of files in the Dropbox dataset folder
    dbx_filenames = set(entry.name for entry in dbx.files_list_folder(dropbox_dataset_folder).entries)

    # Get a list of files in the local dataset folder
    local_filenames = set(filename for filename in os.listdir(local_dataset_folder))

    # Find missing files in the local folder compared to Dropbox
    missing_files = dbx_filenames - local_filenames

    # Download missing files from Dropbox
    for missing_file in missing_files:
        dropbox_file_path = os.path.join(dropbox_dataset_folder, missing_file)
        local_file_path = os.path.join(local_dataset_folder, missing_file)

        # Download the missing file from Dropbox
        dbx.files_download_to_file(local_file_path, dropbox_file_path)
        print(f"Downloaded: {missing_file}")

    print("Dataset update completed.")


### MAIN ###
def main():
    '''Basic command-line application workflow.'''

    # Initial startup.
    if not os.path.exists("dropbox_token.json"):
        download_dataset_from_dropbox()
        subprocess.run(["python", "ai_training_utils.py"])
        print("Model training completed.")
    
    # Load model
    model = load_model('./model_checkpoint')

    # Preparing to loop through the colors.
    colors = {
        'W': 'white',
        'R': 'red',
        'B': 'blue',
        'O': 'orange',
        'G': 'green',
        'Y': 'yellow'
    }
    face_order = ['W', 'R', 'B', 'O', 'G', 'Y']
    colorstrings = {}

    # Get a colorstring for each face of the cube.
    for center_color in face_order:
        face_name = colors[center_color]
        size, edged, colstring, prcsd_img = capture_cube_face(center_color, face_name, model)
        colorstrings[center_color] = (size, edged, colstring, prcsd_img)

        upload_choice = input("\nDo you want to upload the image and data associated with this colorstring to Dropbox? (Y/N): ")
        if upload_choice.lower() == 'y':
            upload_to_dropbox(img=prcsd_img, size=size, edged=edged, colstring=colstring)

    # Concatenate colorstrings to ensure 9 of each color...
    colstring_list = []

    for tup in colorstrings.values():
        colstring_list.append(tup[2])

    checkerstr = ''.join(colstring_list)

    while validate_checkerstring(checkerstr) == False:
        # Last-chance error correction.

        colstring_list2 = []
        print("We're sorry, but something is wrong with your inputs.") 
        print("There is either a duplicate cubelet or you mistyped a correction.")
        print("Reference your physical cube during these corrections.")
        print("Follow the same order as you did before.")

        for colstring in colstring_list:
            print("Original: "+colstring)
            colstring = input("New: ").upper()
            colstring_list2.append(colstring)
        
        checkerstr = ''.join(colstring_list2)


    # Solve the cube using the obtained colorstrings.
    solution = cubesolutions(*colstring_list)
    print("Cube Solution:", solution) 
    print("Be sure to orient your cube so that White is forward, Orange is up, and that Blue is to the right.\nHave fun, and thanks for your time.")

if __name__ == '__main__':
    main()
