import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter, ImageEnhance

### FUNC ###

def redden(im:np.ndarray, amount:int) -> np.ndarray:
    if abs(amount)>=2:
        amount2 = 0.01*amount+1
    else:
        amount2 = amount

    im2 = np.copy(im).astype(np.float64)

    im2[:,:,0] *= amount2

    im2 = np.clip(im2, 0, 255)

    return im2.astype(np.uint8)

def greenify(im:np.ndarray, amount:int) -> np.ndarray:
    if abs(amount)>=2:
        amount2 = 0.01*amount+1
    else:
        amount2 = amount

    im2 = np.copy(im).astype(np.float64)

    im2[:,:,1] *= amount2

    im2 = np.clip(im2, 0, 255)

    return im2.astype(np.uint8)

def blueify(im:np.ndarray, amount:int) -> np.ndarray:
    if abs(amount)>=2:
        amount2 = 0.01*amount+1
    else:
        amount2 = amount

    im2 = np.copy(im).astype(np.float64)

    im2[:,:,2] *= amount2

    im2 = np.clip(im2, 0, 255)

    return im2.astype(np.uint8)

def blur(im:np.ndarray, amount:int) -> np.ndarray:
    amount2 = abs(amount)

    if amount2>5:
        amount2=5

    im2 = Image.fromarray(np.copy(im))
    im2 = im2.filter(ImageFilter.GaussianBlur(amount2))
    im2 = np.array(im2)

    return im2.astype(np.uint8)

def brighten(im:np.ndarray, amount:int) -> np.ndarray:
    if abs(amount)>=2:
        amount2 = 0.01*amount+1
    else:
        amount2 = amount

    im2 = Image.fromarray(np.copy(im))
    im2 = ImageEnhance.Brightness(im2).enhance(amount2)
    im2 = np.array(im2)

    return im2.astype(np.uint8)

def contrast(im:np.ndarray, amount:int) -> np.ndarray:
    if abs(amount)>=2:
        amount2 = 0.01*amount+1
    else:
        amount2 = amount

    im2 = Image.fromarray(np.copy(im))
    im2 = ImageEnhance.Contrast(im2).enhance(amount2)
    im2 = np.array(im2)

    return im2.astype(np.uint8)

def saturate(im:np.ndarray, amount:int) -> np.ndarray:
    if abs(amount)>=2:
        amount2 = 0.01*amount+1
    else:
        amount2 = amount

    im2 = Image.fromarray(np.copy(im))
    im2 = ImageEnhance.Color(im2).enhance(amount2)
    im2 = np.array(im2)

    return im2.astype(np.uint8)

def change_temp(im:np.ndarray, amount:int) -> np.ndarray:
    kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}

    temp = 5000+500*amount

    if temp > 8000:
        temp = 8000
    elif temp < 2000:
        temp = 2000
    im2 = Image.fromarray(np.copy(im))
    
    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    im2 = im2.convert('RGB', matrix)
    im2 = np.array(im2)
    
    return im2.astype(np.uint8)

def parse_file(file_path:str) -> tuple[bool, str, bool, int]:
        try:
            filename = os.path.split(file_path)[1]
            filename = filename[:-4]
            edge, colstring, size, num = filename.split('_')
        
            return (True if edge == 'edge' else False), colstring, (True if size == 'large' else False), int(num)
        except Exception as ex:
            return None, None, None, None

def parse_dataset(file_folder:str, ext:str='.png') -> pd.DataFrame:
    
    records = []
    files = []
    for file_ext in os.listdir(file_folder):
        if file_ext[-4:] == ext:
            file_path = file_folder+"/"+file_ext
            info = parse_file(file_path)
            records.append(info)
            files.append(file_ext)
    
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['edged', 'colorstring', 'large', 'number', 'file']
    df = df.dropna()

    return df

def color_STR_to_INT(col:str) -> str:
    if col == 'W':
        return 5
    elif col == 'O':
        return 1
    elif col == 'G':
        return 3
    elif col == 'R':
        return 0
    elif col == 'B':
        return 4
    elif col == 'Y':
        return 2
    else:
        return None
    
def colstring_to_colors(df:pd.DataFrame) -> pd.DataFrame:
    full_records = []
    for colstring in df['colorstring'].values:
        records = {}
        for idx, char in enumerate(colstring):
            color = color_STR_to_INT(char)
            col_name = 'square_'+str(idx)
            records[col_name] = color
        full_records.append(records)

    df_modified = df.copy(deep=True)

    square_column_names = ['square_'+str(num) for num in range(9)]
    
    df_modified[square_column_names] = pd.DataFrame(full_records)

    df_modified = df_modified.drop(['colorstring'], axis='columns')
    return df_modified

def dataset_to_dataframe(folder_name:str='dataset_imgs') -> pd.DataFrame:
    df = parse_dataset(folder_name)
    df = df.sort_values('number')
    df = df.reset_index()
    df = df.drop(['number', 'index'], axis='columns')
    df = colstring_to_colors(df)
    return df