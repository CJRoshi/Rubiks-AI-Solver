import numpy as np
import pandas as pd
#import pandas_ml as pdml
import cv2
import tensorflow as tf
from keras.utils import to_categorical
import ai_basics
from PIL import Image

TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 360
dataset_folder = 'dataset_imgs/'

col_names = []
for num in range(9):
    col_name = 'square_'+str(num)
    col_names.append(col_name)

data = ai_basics.dataset_to_dataframe()
print(data)

class RubiksCubeFaceDataGen():
    '''
    Data generator for the images in dataset_imgs. Used to train the model.
    '''

    def __init__(self, df):
        self.df = df

    def generate_indices(self):
        # Shuffle df, and create a training set, test set, and validation set.
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df)*TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to*TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

        return train_idx, valid_idx, test_idx
    
    def preprocess_quick(self, img_path):
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0

        return im
    
    def image_read(self, image_idx, in_training, batchsize=10):
        "Generates image batches when using the model."

        images, sizes, edgedes, square_0s, square_1s, square_2s, square_3s, square_4s, square_5s, square_6s, square_7s, square_8s = [], [], [], [], [], [], [], [], [], [], [], []
        while True:
            for idx in image_idx:
                face=self.df.iloc[idx]

                size = [1 if face['large'] else 0]
                edge = [1 if face['edged'] else 0]
                square_0 = face['square_0']
                square_1 = face['square_1']
                square_2 = face['square_2']
                square_3 = face['square_3']
                square_4 = face['square_4']
                square_5 = face['square_5']
                square_6 = face['square_6']
                square_7 = face['square_7']
                square_8 = face['square_8']

                file = 'dataset_imgs/'+face['file']

                im = self.preprocess_quick(file)

                sizes.append(to_categorical(size, 2))
                edgedes.append(edge)
                square_0s.append(to_categorical(square_0, 6))
                square_1s.append(to_categorical(square_1, 6))
                square_2s.append(to_categorical(square_2, 6))
                square_3s.append(to_categorical(square_3, 6))
                square_4s.append(to_categorical(square_4, 6))
                square_5s.append(to_categorical(square_5, 6))
                square_6s.append(to_categorical(square_6, 6))
                square_7s.append(to_categorical(square_7, 6))
                square_8s.append(to_categorical(square_8, 6))
                images.append(im)

                # YIELD
                if len(images) >= batchsize:
                    yield np.array(images), [np.array(sizes), np.array(edgedes), 
                                             np.array(square_0s), np.array(square_1s), np.array(square_2s), np.array(square_3s), 
                                             np.array(square_4s), np.array(square_5s), np.array(square_6s), np.array(square_7s),
                                             np.array(square_8s)]
                    images, sizes, edgedes, square_0s, square_1s, square_2s, square_3s, square_4s, square_5s, square_6s, square_7s, square_8s = [], [], [], [], [], [], [], [], [], [], [], []

            if not in_training:
                break

datagen = RubiksCubeFaceDataGen(data)
train_idx, valid_idx, test_idx = datagen.generate_indices()
                    


