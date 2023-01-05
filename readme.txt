~~~~~~~~~~~~~~~~~

BRANCH OVERVIEW

~~~~~~~~~~~~~~~~~

gui_app: This branch is dedicated to developing a gui/application for the Rubik's Cube Solver. 

It must:

- Offer a way to "set" which side is what color, and what the color of the top side is
- Use the functions created in crop_image and white_balance to analyze each image of the cube
- Feed these into solve_cube's functions
- Present the solutions in human-readable format, perhaps by using images, or just leave a setting for notation.


crop_image: This branch is dedicated to developing a function (or set thereof) to prepare an area
            of the image for white_balance and cube_solve.

It must:

- Crop the center portion of the camera's view into a 3x3 area
- Attempt to correct for perspective by way of edge detection
- Feed the results into white_balance

white_balance: This branch is dedicated to developing a set of functions to white-balance the image prepared
            by crop_image. It will also check for what the color of these regions are once this is accomplished.
            Results will be fed into cube_solve.

It must:

- White-balance the image
- Feed results (preferably by way of list or array) into average_color

average_color: This branch is dedicated to finsing the median (or mode) color of a region from white_balance.

It must:

- Return the median color of a region. If this cannot be determined, use the mode instead.
- Feed results directly into cube_solve by list or array.

cube_solve: Exactly what it says on the tin! This branch will develop one function dedicated to solving the cube
            using kociemba or some other method. 

It must:

- Solve the Rubik's cube (optionally, in regular, cross, AND cube-in-cube patterns)
- Return a list of moves for gui_app to interpret
- Contain a verification method so that the code is legitimate!
    + Not joking about this, I want it to be so that this thing doesn't run if a junk file is missing :)

 
