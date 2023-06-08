import time
import base64
import cv2
import numpy as np
import os
from PIL import Image
from crop_image import guidelines
from keras.models import load_model


# Load AI model.
model = load_model('./model_checkpoint')

'''____________________________ FUNC __________________________________'''
import requests

def get_folder_filenames(owner, repo, folder_path, token):
    # GitHub repository information
    base_url = "https://api.github.com"
    endpoint = f"/repos/{owner}/{repo}/contents/{folder_path}"

    # Prepare the request headers
    headers = {
        "Authorization": token,
        "Accept": "application/vnd.github.v3+json"
    }
    print(base_url+endpoint)
    # Send GET request to fetch the contents of the folder
    response = requests.get(base_url + endpoint, headers=headers)

    if response.status_code == 200:
        # Extract the filenames from the response
        filenames = [item["name"] for item in response.json() if item["type"] == "file"]
        return filenames
    else:
        print("Failed to fetch folder contents:", response.text)
        return []

def get_next_available_number(filenames):
    used_numbers = set()
    for filename in filenames:
        parts = filename.split("_")
        if len(parts) == 4 and parts[-1].endswith(".png"):
            num = parts[-1].split(".")[0]
            used_numbers.add(num)

    next_number = 1
    while str(next_number) in used_numbers:
        next_number += 1

    return next_number

def upload_to_github(img, size, edged, colorstring):

    img = Image.fromarray(img)

    owner = "Fireclaw29121"
    repo = "Rubiks-AI-Solver"
    folder_path = "dataset_imgs"
    token = os.environ.get('GITHUB_TOKEN')
    print(token)

    filenames = get_folder_filenames(owner, repo, folder_path, token)
    next_num = get_next_available_number(filenames)

    filename = edged+"_"+colorstring+"_"+size+"_"+str(next_num)+".png"
    # Save the image as a PNG file
    img.save(filename, "PNG")
    

    # Read the photo file
    with open(filename, "rb") as file:
        photo_data = file.read()

    # Create a new file on GitHub
    headers = {
        "Authorization": token,
        "Content-Type": "application/json",
    }
    create_file_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder_path}"
    print(create_file_url)
    payload = {
        "branch" : "main",
        "message": "Upload new photo to dataset",
        "content": base64.b64encode(photo_data).decode("utf-8"),  # Convert binary data to base64-encoded string
    }
    response = requests.put(create_file_url, headers=headers, json=payload)
    
    if response.status_code == 201:
        print("File uploaded successfully!")
    else:
        print("Failed to upload file:", response.text)

    os.remove(filename)

def analyze_photo(img:cv2.Mat, model) -> tuple[str, str, str]:

    ### IMAGE PROCESSING ###
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgheight, imgwidth, depth = np.shape(img)

    if imgwidth >= imgheight:
        larger_side = imgwidth
        smaller_side = imgheight

        border_top  = (smaller_side)//8
        border_side = (larger_side)//2-3*border_top
        #true_center_width = 4*border_top (artifact from segmentation-based approach)
        center_width = 6*border_top

    elif imgwidth < imgheight:
        larger_side = imgheight
        smaller_side = imgwidth

        border_side  = (smaller_side)//8
        border_top = (larger_side)//2-3*border_side
        #true_center_width = 4*border_side
        center_width = 6*border_side

    # Crop the img to the above dimensions.
    img_cropped = np.copy(img)
    img_cropped = img_cropped[border_top:border_top+center_width, border_side:border_side+center_width]

    '''
    ### FILE CLASSIFICATION ###
    edge = "edge_" if "y" in input("Is this an edged cube? (Y/N)\n").lower() else "noedge_"
    print("Write the colors on this face in sequence, such as 'WWWWWWWWW' for all-white or 'ROYGBWBGY'.")
    print("R is RED\nO is ORANGE\nY is YELLOW\nG is GREEN\nB is BLUE\nW is WHITE\n")
    colstring = input().upper() + "_"
    small = "large_" if "y" in input("Is this a closeup? (Y/N)\n").lower() else "small_"
    cv2.imwrite((folder+edge+colstring+small+str(image_num)+'.png'), img)
    image_num+=1
    '''
    
    # Shape image to model's expectations...
    img_cropped = Image.fromarray(img_cropped)
    img_cropped = img_cropped.resize((360,360))
    img_cropped = np.asarray(img_cropped)
    img_cropped = np.reshape(img_cropped, (1,360,360,3))


    ### AI PREDICTS ###
    predictions = model.predict(img_cropped)

    results = [tensor[0] for tensor in predictions]

    if int(results[0][0])== 1:
        size = 'large'
        #print("AI predicts this is a close-up.")
    else:
        size = 'small'
        #print("AI predicts this is not a close-up.")

    if int(results[1][0]) == 1:
        edged = 'edged'
        #print("AI predicts this is an edged cube.")
    else:
        edged = 'noedge'
        #print("AI predicts this is an edgeless cube.")

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

    #print("AI predicts colors: "+colstring)
    return size, edged, colstring

'''________________________________ MAIN _______________________________'''

# Initialize the camera object.
cam = cv2.VideoCapture(1)
result, img = cam.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Time delay to allow the shutter to autofocus.
time.sleep(0.5)
image_num=75

while result:
    # Capture video frame by frame
    result, img = cam.read()
    

    # Place guidelines on a copy of the img.
    img_guidelines = guidelines(img)
    
    # Display the resulting copy.
    #cv2.imshow(('VIDEO FEED -- Captured '+str(image_num)+' images.'), img_guidelines)

    cv2.imshow(('VIDEO FEED'), img_guidelines) 
    # Use Q to quit.
    if cv2.waitKey(81)  == ord('q'):
        break

    # Use P to capture and process images.
    if cv2.waitKey(80) == ord('p'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        size, edged, colstring = analyze_photo(img, model)
        print(colstring)
        colstring = input("CORRECT IT.\n")
        size = input("'small' or 'large'. There is no other option.\n")
        edged = input("'edge' or 'noedge', that is the question.\n")
        upload_to_github(img, size, edged, colorstring=colstring)

        cv2.destroyAllWindows()
        time.sleep(0.75)

    # After the loop, release the camera object.
cam.release()

    # Destroy all windows.
cv2.destroyAllWindows()