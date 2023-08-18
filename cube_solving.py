"""
CUBE SOLVING
Author: Nino R, Blanca S, Alekhya P, Caden B.
Date: 6/28/2023
This code uses the solver from https://github.com/muodov/kociemba and hooks it up to our AI.
Due to errors when pip-installing the module, its source folder is present in the repo. 
Big thanks to muodov for allowing the code to be used under the GNU GPL.
"""

### IMPORT ###
from kociemba import solve



### FUNC ###

def cubesolutions(colstrW:str, colstrR:str, colstrB:str, colstrO:str, colstrG:str, colstrY:str) -> str:
    '''Takes in all the colorstrings for the faces and processes them to return the solution for the cube.'''
    # Blank, unmapped cube.
    cubestr = ["X"]*54

    # White/FRONT
    cubestr[18:27] = colstrW[0], colstrW[1], colstrW[2], colstrW[3], colstrW[4], colstrW[5], colstrW[6], colstrW[7], colstrW[8]

    # Red/DOWN
    cubestr[27:36] = colstrR[0], colstrR[1], colstrR[2], colstrR[3], colstrR[4], colstrR[5], colstrR[6], colstrR[7], colstrR[8]

    # Blue/RIGHT (rotate left 90 degrees)
    cubestr[9:18] = colstrB[2], colstrB[5], colstrB[8], colstrB[1], colstrB[4], colstrB[7], colstrB[0], colstrB[3], colstrB[6]

    # Orange/UP (roate 180 degrees)
    cubestr[0:9] = colstrO[8], colstrO[7], colstrO[6], colstrO[5], colstrO[4], colstrO[3], colstrO[2], colstrO[1], colstrO[0]

    # Green/LEFT (rotate right 90 degrees)
    cubestr[36:45] = colstrG[6], colstrG[3], colstrG[0], colstrG[7], colstrG[4], colstrG[1], colstrG[8], colstrG[5], colstrG[2]

    # Yellow/BACK (rotate right 90 degrees)
    cubestr[45:54] = colstrY[6], colstrY[3], colstrY[0], colstrY[7], colstrY[4], colstrY[1], colstrY[8], colstrY[5], colstrY[2]

    cubestr = ''.join(cubestr).upper()

    # Change colors to their relative positions.
    cubestr = cubestr.replace('R', 'D').replace('W', 'F').replace('B', 'R').replace('O', 'U').replace('Y', 'B').replace('G', 'L')

    # Return the solution.
    return solve(cubestring=cubestr)

### MAIN ###

def main():
    # For testing purposes... 
    colstrW = input("White side: ")
    colstrR = input("Red side: ")
    colstrB = input("Blue side: ")
    colstrO = input("Orange side: ")
    colstrG = input("Green side: ")
    colstrY = input("Yellow side: ")


    # Checking that center colors are correct...

    if colstrW[4] != 'W' or len(colstrW)!=9:
        print("The first side isn't the white side!")
    if colstrR[4] != 'R' or len(colstrR)!=9:
        print("The second side isn't the red side!")
    if colstrB[4] != 'B' or len(colstrB)!=9:
        print("The third side isn't the blue side!")
    if colstrO[4] != 'O' or len(colstrO)!=9:
        print("The fourth side isn't the orange side!")
    if colstrG[4] != 'G' or len(colstrG)!=9:
        print("The fifth side isn't the green side!")
    if colstrY[4] != 'Y' or len(colstrY)!=9:
        print("The sixth side isn't the yellow side!")

    print(cubesolutions(colstrW, colstrR, colstrB, colstrO, colstrG, colstrY))

if __name__ == '__main__':
    main()