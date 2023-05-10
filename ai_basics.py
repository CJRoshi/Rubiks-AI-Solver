import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf

### FUNC ###

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

def color_short_to_long(col:str) -> str:
    if col == 'W':
        return 6
    elif col == 'O':
        return 2
    elif col == 'G':
        return 4
    elif col == 'R':
        return 5
    elif col == 'B':
        return 4
    elif col == 'Y':
        return 3
    else:
        return None
    
def colstring_to_colors(df:pd.DataFrame) -> pd.DataFrame:
    full_records = []
    for colstring in df['colorstring'].values:
        records = {}
        for idx, char in enumerate(colstring):
            color = color_short_to_long(char)
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