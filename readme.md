# **Welcome to RubiksNet!**

RubiksNet is a small project from the North Andover High School Engineering Club that uses Deep Learning (in the form of a Convolutional Neural Network) to try to solve a Rubik's Cube from *your* image input.

## Installation Guide ##

> It is strongly recommended that you install RubiksNet on an external drive due to its size. We recommend having a GPU to train the model efficiently. The minimum requirement is at least 8GB of RAM.
>
### **When you download RubiksNet:**

1. Open Powershell or cmd.exe
2. Change directory (cd) to your install
3. Run `pip install -r requirements.txt`.
4. Allow the python libraries to install. This may take a while.
5. Run `RubiksNetApp.py` (And shortcut it for later use if you'd like)
6. **Have fun!**

## Project Guide ##

### 1. **Purpose**
* The purpose of this project was initially to explore the basics of Python with our members using a fun and useful project. Once we covered the basics we decided to explore AI.
* The purpose of the code itself is to implement a very simple application flow, which can be summarized like this:

```
for each face of a Rubik's Cube:
    capture_face() # Webcam
    analyze_face() # With AI
    correct_predictions() # User input

solve_cube(corrected_predicitons)
```

### 2. **Capturing Images**
* Capturing images was done with `OpenCV`.
* `OpenCV` provides several useful methods for image analysis, and we tried to use some in `image_segmentation.py`. However, they were not too effective.
* This was why we switched to our AI-based approach.
* When capturing images, we have a video label in the GUI that is updated with webcam output.
* Individual frames can be easily captured and processed by RubiksNet.

### 3. **RubiksNet: The Model**
* RubiksNet predicts several attributes on an image. It predicts:
    * the color of each square;
    * the size of the Rubik's Cube in the image (does it take up almost the whole image or not);
    * and whether or not the cube is edgeless.
* To achieve this, we had to do the following:
    1. We had to gather data. This data exists online in a Dropbox folder, and is downloaded locally to your device on open. We gathered this data ourselves using Nino's Surface Pro 7.
        * The data is a collection of `360x360` RGB images. Their filenames contain the labeling data for training. 
    2. We created several functions in `ai_data_utils.py` that assist in turning this dataset directory into a `pandas DataFrame`. 

    3. Then, we had to create the model, which took several steps.
        1. We decided on using a **Convolutional Neural Network** because of its relative ease of implementation and its power in image analysis.
        2. We implemented the model, which can be seen in `ai_training_utils.py`. 
            * Initial results were okay, but not excellent.
            * Thus, we decided to combat the problem of our small dataset by introducing *data augmentation.*
        3. Data Augmentation was implemented, allowing the model to give itself more data to train off of.
        4. The model has these hyperparameters at current: 
            * Train-Test Split = `0.7`
            * *Training* batch size = `8`
            * *Validation* batch size = `16`
            * Learning Rate = `1e-5, 1*10^-5`
            * Number of epochs = `25`
            * Total dataset size (because data augmentation allows for "larger" datasets than just the images provided) = `len(dataset_imgs [dir])`
        > Feel free to edit `ai_training_utils.py` and re-run it to train the model again with new params.
        >
        5. We chose these hyperparameters to run on a ***bare-minimum*** system, am 8GB RAM Surface Pro 7 with ~10 GB of disk space free. This also introduced some limits in accuracy.
        6. We strongly encourage users to tinker with adding or removing neurons, convolutional layers, and changing hyperparameters to improve results. 

### 4. **Solving Cubes**

* Cube solving was relatively simple to implement because other authors have already created algorithms for it. Implementation can be found in `cube_solving.py`.
* This code uses [muodov's Kociemba solver](https://github.com/muodov/kociemba) and hooks it up to our AI.
    * Due to errors when pip-installing the module, its source folder is present in the repo. 
    * Big thanks to muodov for allowing the code to be used under the `GNU GPL`, the licesnse for this project.

### 5. **The GUI/RubiksNet App**

* This was tricky to implement. The [application flow described](#1.-Purpose) is essentially the same, though we'd like to highlight some specifics of the app.
    * We devised a system to make it easier for the user to correct color predictions.
    * We call this system **ColorSquare**.
    * It creates a grid of ColorSquares that behave as follows:
    ```
    def change_color(self, button:tk.Button, color_buttons:list[tk.Button], center_color:str):
            '''
            Inputs:
        
            button (tk.Button):                 The target ColorSquare.

            color_buttons (list[tk.Button]):    A list of all ColorSquares. Used to determine whether or not the center color is valid.

            center_color (str):                 A single letter of "WROYGB". 
                                                The current center color being requested by submit_corrections().

            This function takes the current ColorSquare and examines its color (text). Then, it uses the color_info dictionary
            to modify the button's appearance and text. This function also checks, whenever a button is clicked, if the center button matches the expected center color, and sets the states of next_button and upload_button to NORMAL if true. 
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
    ```
    * This system is easy for the user, as they just have to click the buttons repeatedly to cycle to the right color.
    * We also gave users the option to upload their images to RubiksNet's dataset via `upload_to_dropbox_button`, though a user can just as easily press `next_face_button` to continue without uploading.

## End Project Guide ##

#### That's all for now. Have fun solving! :)
