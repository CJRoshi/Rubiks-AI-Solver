"""
GUI APP / RUBIKSNET APP
Author: Nino R
Date: 8/19/2023
This code implements the RubiksNet App, and its GUI.
"""
import os
import subprocess
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk

from cmdlineapp import (analyze_photo, download_dataset_from_dropbox,
                        update_dataset_from_dropbox, upload_to_dropbox,
                        validate_checkerstring)
from cube_solving import cubesolutions
from image_segmentation import guidelines


class RubiksApp:
    '''
    Class that defines the GUI for the RubiksNet App, as well as its workflow.
    '''
    
    def __init__(self, root):
        '''
        __init__ function. This function sets up the main application's notebook flow and creates the start page.
        It also initializes several variables.
        '''
        # Main Window
        self.root = root
        self.root.title("RubiksApp")
        self.root.state('zoomed')
        self.root.wm_attributes("-topmost", 1)

        # Schedule a function to disable topmost priority after some time
        self.root.after(50, self.disable_topmost)

        # Start Page Structure
        self.notebook = ttk.Notebook(root)
        self.notebook.pack()

        self.capture_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.capture_frame, text="RubiksNet App v1.0")
        

        # Add a button to start the color capturing process
        start_capture_button = tk.Button(self.capture_frame, text="Start Solving!", command=self.start_color_capturing)
        start_capture_button.pack(pady=20)

        # Dictionary containing all the info for how ColorSquares are meant to look.
        self.color_info = {
            "W": {"next_color": "R", "button_background": "white", "button_active_background": "gray",
                  "button_text_color": "black"},
            "R": {"next_color": "O", "button_background": "red", "button_active_background": "darkred",
                  "button_text_color": "white"},
            "O": {"next_color": "Y", "button_background": "orange", "button_active_background": "darkorange",
                  "button_text_color": "black"},
            "Y": {"next_color": "G", "button_background": "yellow", "button_active_background": "gold",
                  "button_text_color": "black"},
            "G": {"next_color": "B", "button_background": "green", "button_active_background": "darkgreen",
                  "button_text_color": "white"},
            "B": {"next_color": "W", "button_background": "blue", "button_active_background": "darkblue",
                  "button_text_color": "white"}
        }

        # Load RubiksNet from its files.
        self.model = load_model('./model_checkpoint')

        # At this step, also check for new dataset files.
        update_dataset_from_dropbox()

        # Various dummy initial variables that are used with analyze_photo in processing.
        self.prcsd_img = None
        self.colstring = None
        self.size = None
        self.edged = None

        # Dummy initial variables relating to the correction section.
        self.prcsd_img_tk = None
        self.next_button = None
        self.center_color = "W"
        self.size_var = tk.StringVar(value="large")
        self.edged_var = tk.StringVar(value="edge")

        # Initialization of variables for looping
        self.color_index = 0
        self.color_order = "WRBOGY"  # Order of colors to capture and correct
        self.corrected_colorstrings = []  # To store the corrected colorstrings
        self.root.mainloop()

    def disable_topmost(self):
        '''A function that disables the app's window priority.'''
        self.root.wm_attributes("-topmost", False)  # Disable topmost after 3 seconds

    def update_video_feed(self):
        '''
            A function that updates the video Label seen in capture_cube_face. 
            Stores cam's primary output in self.img_rgb.
        '''
        ret, img = self.cam.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.img_rgb = img_rgb  # Store img_rgb as an instance variable
        
        img_guidelines = guidelines(img_rgb.copy())
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_guidelines))
        self.video_label.config(image=img_tk)
        self.video_label.img = img_tk
        self.root.after(10, self.update_video_feed)

    def change_color(self, button:tk.Button, color_buttons:list[tk.Button], center_color:str):
        '''
        Inputs:
    
        button (tk.Button):                 The target ColorSquare.

        color_buttons (list[tk.Button]):    A list of all ColorSquares. Used to determine whether or not the center color is valid.

        center_color (str):                 A single letter of "WROYGB". 
                                            The current center color being requested by submit_corrections().

        This function takes the current ColorSquare and examines its color (text). Then, it uses the color_info dictionary
        to modify the button's appearance and text. This function also checks, whenever a button is clicked, if the center button
        matches the expected center color, and sets the states of next_button and upload_button to NORMAL if true. 
        Otherwise, the state is set to DISABLED.
        '''

        # Read current button info, get new info for cycling
        current_color = button["text"]
        next_color = self.color_info[current_color]['next_color']
        next_color_info = self.color_info[next_color]

        # Set the button's new attributes.
        button.config(text=next_color, bg=next_color_info["button_background"], 
                      activebackground=next_color_info["button_active_background"], 
                      fg=next_color_info["button_text_color"])

        # Check if the corrected center color matches the expected center color
        if color_buttons[4]["text"] == center_color:
            self.next_button.config(state=tk.NORMAL)
            self.upload_button.config(state=tk.NORMAL)
        else:
            self.next_button.config(state=tk.DISABLED)
            self.upload_button.config(state=tk.DISABLED)

    def submit_corrections(self, prcsd_img:np.ndarray, colstring:str, size:str, edged:str, center_color:str):
        '''
        Inputs:
    
        prcsd_img: (np.ndarray):    A processed version of the webcam input from capture_cube_face generated by analyze_photo().

        colstring (str):            A 9 letter string with the colors of one face of the Rubik's Cube, such as "WWWWWWWWW" for
                                    the white face of the solved cube, generated by analyze_photo().

        size (str):                 Either "large" or "small". A judgement made by RubiksNet in analyze_photo() 
                                    of how much space the Rubik's Cube takes up in prcsd_img.

        edged (str):                Either "edge" or "noedge". A judgement by RubiksNet made in analyze_photo()
                                    of whether or not the cube is an edgeless cube.

        center_color (str):         The current center color being requested by capture_image() and capture_cube_face().

        This function takes the results of analyze_photo (and RubiksNet by proxy) and displays them in a Submit Corrections window.
        In order, it does the following:

        It initializes prcsd_img for display in a copy.
        It creates a grid of nine ColorButtons with initial values from colstring, whose behavior is described in change_color().
        It uses size and edged to initialize the states of four radiobuttons, two for each attribute.
        It initializes two buttons, "Next Face" and "Upload to Dropbox". Their behavior is described in their respective functions.
        '''
        # Reading the new values over the old.
        self.prcsd_img = prcsd_img
        self.colstring = colstring
        self.size = size
        self.edged = edged
        self.center_color = center_color

        # Initialize Correction Frame
        self.corrections_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.corrections_frame, text="Correct RubiksNet's Predictions")

        # Displayable version of processed image
        self.prcsd_img_tk = ImageTk.PhotoImage(image=Image.fromarray(prcsd_img))
        self.prcsd_img = prcsd_img        

        prcsd_img_label = tk.Label(self.corrections_frame, image=self.prcsd_img_tk)
        prcsd_img_label.pack()

        # ColorSquares variables and positioning
        color_infodict = self.color_info # Used only for initialization here.
        color_buttons_frame = tk.Frame(self.corrections_frame)
        color_buttons_frame.pack()

        self.color_buttons = []

        # Colorsquares initialized
        for idx, color in enumerate(colstring):
            button_info = color_infodict[color]
            button_color = button_info["button_background"]
            color_button = tk.Button(color_buttons_frame, text=color, width=5, height=2,
                                    bg=button_color,
                                    activebackground=button_info["button_active_background"],
                                    highlightthickness=2,
                                    fg=button_info["button_text_color"])
            self.color_buttons.append(color_button)
            row = idx // 3
            col = idx % 3
            color_button.grid(row=row, column=col, padx=0, pady=0)
            # Buttons given functionality here.
            color_button.config(command=lambda button=color_button: self.change_color(button, self.color_buttons, self.center_color))

        self.size_var.set(size)  # Set the initial value for size_var
        self.edged_var.set(edged)  # Set the initial value for edged_var

        # Size Radiobuttons
        size_radiobutton_large = tk.Radiobutton(self.corrections_frame, text="Large", variable=self.size_var, value="large")
        size_radiobutton_small = tk.Radiobutton(self.corrections_frame, text="Small", variable=self.size_var, value="small")
        size_radiobutton_large.pack()
        size_radiobutton_small.pack()

        # Edged radiobuttons
        edged_radiobutton = tk.Radiobutton(self.corrections_frame, text="Edge", variable=self.edged_var, value="edge")
        edgeless_radiobutton = tk.Radiobutton(self.corrections_frame, text="No Edge", variable=self.edged_var, value="noedge")
        edged_radiobutton.pack()
        edgeless_radiobutton.pack()

        # "Next Face" button
        self.next_button = tk.Button(self.corrections_frame, text="Next Face", command=self.next_face_button)
        if self.colstring[4] == self.center_color:
            self.next_button.config(state=tk.NORMAL)
        else:
            self.next_button.config(state=tk.DISABLED)
        self.next_button.pack()


        # "Upload to Dropbox" button
        self.upload_button = tk.Button(self.corrections_frame, text="Upload Corrections to Dropbox", command=self.upload_to_dropbox_button, bg="darkblue", fg="white", activebackground="blue")
        if self.colstring[4] == self.center_color:
            self.upload_button.config(state=tk.NORMAL)
        else:
            self.upload_button.config(state=tk.DISABLED)
        self.upload_button.pack()


        self.notebook.select(self.corrections_frame)  # Switch to the corrections frame after initializing

    def upload_to_dropbox_button(self):
        '''
        This button reads the current state of all ColorSquares 
        and the StringVars associated with the size and edged radiobuttons 
        to gather metadata to upload with prcsd_img to the RubiksNet Dropbox, which is
        handled by the upload_to_dropbox() function.

        It also handles updating the index of the loop in capture_all_colors() 
        and adding a colorstring to the list of colorstrings. 
        If that loop is "finished", it proceeds to the cube solving method.
        '''

        # Gather metadata.
        updated_colstring = "".join([button["text"] for button in self.color_buttons])
        size = self.size_var.get()
        edged = self.edged_var.get()

        # Call the upload_to_dropbox function with the updated values
        upload_to_dropbox(img=self.prcsd_img, size=size, edged=edged, colstring=updated_colstring)
        self.corrected_colorstrings.append(updated_colstring) # Append colstring to list

        # Move to the next frame
        self.corrections_frame.destroy()

        # Update loop logic
        self.color_index += 1
        if self.color_index < len(self.color_order):
            # Capture the next color face
            self.capture_cube_face(self.color_order[self.color_index])
        else:
            # All color faces have been captured
            checkerstr = ''.join(self.corrected_colorstrings)
            if not validate_checkerstring(checkerstr):
                self.run_last_second_corrections()  # Run last-second corrections if needed
            else:
                self.run_cube_solver(self.corrected_colorstrings)  # Proceed to cube solving right away

    def next_face_button(self):
        '''
        This button reads the current state of all ColorSquares to create a colorstring 
        for the list of colorstrings.

        It also handles updating the index of the loop in capture_all_colors() 
        and adding a colorstring to the list of colorstrings. 
        If that loop is "finished", it proceeds to the cube solving method.
        '''
        # Gather ColorSquare states
        updated_colstring = "".join([button["text"] for button in self.color_buttons])
        self.corrected_colorstrings.append(updated_colstring) # Append colstring to list

        # Move to the next frame
        self.corrections_frame.destroy()

        # Update loop logic
        self.color_index += 1
        if self.color_index < len(self.color_order):
            # Capture the next face
            self.capture_cube_face(self.color_order[self.color_index])
        else:
            # All color faces have been captured...
            checkerstr = ''.join(self.corrected_colorstrings)
            if not validate_checkerstring(checkerstr):
                self.run_last_second_corrections()  # Run last-second corrections if needed
            else:
                self.run_cube_solver(self.corrected_colorstrings)  # Proceed to cube solving right away


    def start_color_capturing(self):
        '''Just the command behind the "Start" button.'''
        self.capture_all_colors()

    def capture_all_colors(self):
        "A command that's used in the loop of capturing colors. Will always start with index 0 (White)."
        self.capture_cube_face(self.color_order[self.color_index])

    def capture_cube_face(self, center_color:str):
        '''
        Inputs:

        center_color (str): The current center color being requested. Used for instruction and to help guide submit_corrections().

        This function sets up a basic image capture window and a button to capture images, along with user instructions.
        '''
        # Destroy the previous frame, set up the new one.
        self.capture_frame.destroy()
        self.capture_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.capture_frame, text="Capture A Face on Your Cube")

        # Give the user instructions depending on what color they're looking for.
        self.instruction_label = tk.Label(self.capture_frame)
        if center_color == 'W':
            self.instruction_label.config(text="Position the cube so that the white center is forward and the red center is down.")
        elif center_color == 'R':
            self.instruction_label.config(text="Rotate up to red.")
        elif center_color == 'B':
            self.instruction_label.config(text="Rotate left to blue.")
        elif center_color == 'O':
            self.instruction_label.config(text="Rotate left to orange.")
        elif center_color == 'G':
            self.instruction_label.config(text="Rotate left again to green.")
        elif center_color == 'Y':
            self.instruction_label.config(text="Rotate up one last time to yellow.")
        self.instruction_label.pack()

        # Video label initialized, updates
        self.video_label = tk.Label(self.capture_frame)
        self.video_label.pack(pady=10)

        self.cam = cv2.VideoCapture(0) # Front Webcam by default.
        self.update_video_feed()

        # Capture Image button
        self.capture_button = tk.Button(self.capture_frame, text="Capture Image", command=lambda: self.capture_image(center_color))
        self.capture_button.pack(pady=10)

    def capture_image(self, center_color:str):
        '''
        Inputs:

        center_color (str): The current center color being requested. Used for instruction and to help guide submit_corrections().

        This function sets up the basic process that occurs after capturing an image, that being to process it and correct it.
        '''

        if self.img_rgb is not None:
            # Process the image and generate predictions using analyze_photo
            size, edged, colstring, prcsd_img = analyze_photo(self.img_rgb, self.model)

            # Correct the image with submit_corrections()
            self.submit_corrections(prcsd_img, colstring, size, edged, center_color)

            # Turn off camera, destroy previous frame, move to next frame.
            self.cam.release()
            self.capture_frame.destroy()
            self.notebook.select(self.corrections_frame)

    def run_last_second_corrections(self):
        '''
        A function used after taking all the images of the Rubik's Cube used only if 
        the colorstrings that would be fed to cube solving do not make an actual cube.

        It sets up a frame where the user manually enters colorstrings as corrections and
        does so until a valid cube is formed.
        '''

        # Check the string.
        checkerstr = ''.join(self.corrected_colorstrings)

        if not validate_checkerstring(checkerstr):

            # Setup last-second frame
            self.last_second_corrections_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.last_second_corrections_frame, text="Last-Second Corrections")

            instructions_label = tk.Label(self.last_second_corrections_frame, text="Re-enter the colorstrings manually, following the exact order of faces you used in collecting them.")
            instructions_label.pack()

            # Generate the labels and entry fields iteratively
            self.colstring_entries = []

            for colstring in self.corrected_colorstrings:
                label = tk.Label(self.last_second_corrections_frame, text=colstring)
                label.pack()
                entry = tk.Entry(self.last_second_corrections_frame)
                self.colstring_entries.append(entry)
                entry.pack()

            # Submit Button
            submit_button = tk.Button(self.last_second_corrections_frame, text="Submit Corrections",
                                    command=self.check_and_submit_last_second_corrections)
            submit_button.pack()
        else:
            # Just in case it goes off by accident for some reason, this is placed here as a catch-all.
            self.run_cube_solver(self.corrected_colorstrings)
            self.last_second_corrections_frame.destroy()

    def check_and_submit_last_second_corrections(self):
        '''
        A function that is responsible for checking the user's last second manual corrections
        and either handling a repeat or going to cube-solving if the corrections are valid.
        '''
        # Gather the entries from the last second corrections frame and perform a check
        colstring_list2 = []
        for entry in self.colstring_entries:
            colstring_list2.append(entry.get().upper())
        checkerstr = ''.join(colstring_list2)

        if validate_checkerstring(checkerstr):
            # Proceed to cube solving
            self.run_cube_solver(colstring_list2)  
            self.last_second_corrections_frame.destroy()
        else:
            # Clear the entries and try again.
            self.clear_last_second_corrections_entries()

    def clear_last_second_corrections_entries(self):
        '''
        Clears all entries on the last second corrections frame.
        '''
        for entry in self.colstring_entries:
            entry.delete(0, tk.END)

    def run_cube_solver(self, colstring_list:list[str]):
        '''
        Inputs:

        colstring_list (list[str]): A list of all 6 colorstrings on the Rubuk's cube. Fed into cubesolutions() to get a cube solution.

        Sets up a solutions frame and displays the solution to the cube.
        '''
        # Set up solution frame
        solution_frame = ttk.Frame(self.notebook)
        self.notebook.add(solution_frame, text="Yay!")

        # Get the cube's solution.
        solution = cubesolutions(*colstring_list)
        solution_label = tk.Label(solution_frame, text="Cube Solution:")
        solution_label.pack()

        solution_text = tk.Text(solution_frame, wrap=tk.WORD, height=10, width=40)
        solution_text.insert(tk.END, solution)
        solution_text.pack()

        # Orientation of the cube may not be "standard"; this is the orientation for the solver
        instructions_label = tk.Label(solution_frame,
                                    text="Orient your cube so that White is forward, Orange is up, and Blue is to the right. Enjoy solving!")
        instructions_label.pack()

def check_initial_startup():
    '''
    This method checks to see if this is the initial startup and if so,
    sets up the dataset and the model.
    '''
    # Check if token.json exists.
    # If token.json doesn't exist, download dataset and train model
    if not os.path.exists("dropbox_token.json"):
        setup_dependencies()
    else:
        # Run the app.
        app = tk.Tk()
        rubiks_app = RubiksApp(app)
        app.mainloop()

def setup_dependencies():
    ''' A function that downloads the dataset and trains the model.'''

    # Download the dataset from Dropbox.
    download_dataset_from_dropbox()

    # Create a separate tkinter window for training progress.
    train_window = tk.Tk()
    train_window.title("Training Progress")
    
    # Instructions for the user.
    model_training_label = tk.Label(train_window, text="Model in training... this may take a while.\nKeep your computer awake while the model trains.")
    model_training_label.pack()

    # Live text stream using subprocess, runs the AI training file.
    model_training_text = tk.Text(train_window, wrap=tk.WORD, state=tk.DISABLED)
    model_training_text.pack()

    # Set up subprocess.
    process = subprocess.Popen(["python", "ai_training_utils.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    def update_training_output():
        '''A function that updates the stdout stream so the user can see the model's training progress.'''

        # Read a line
        line = process.stdout.readline()
        if line:
            # Update the window, call the function to update again
            model_training_text.config(state=tk.NORMAL)
            model_training_text.insert(tk.END, line)
            model_training_text.see(tk.END)
            model_training_text.config(state=tk.DISABLED)
            model_training_text.update_idletasks()
            train_window.after(10, update_training_output)
        else:
            # Training process is finished, destroy train_window
            train_window.destroy()

            # Run the app.
            app = tk.Tk()
            rubiks_app = RubiksApp(app)
            app.mainloop()

    # Update the window, this runs recursively.
    update_training_output()
    train_window.mainloop()

if __name__ == "__main__":
    # Check the initial startup, which runs the app regardless of flow.
    check_initial_startup()
