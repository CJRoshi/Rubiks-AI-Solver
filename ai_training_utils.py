"""
AI TRAINING
Author: Nino R.
Date: 6/14/2023
This code implements a way to train the AI on the images in the dataset, taking advantage of the data utilities in ai_data_utils.py. 
Much of the code here was based on code from https://github.com/rodrigobressan's Face2Data project, which helped me get a grasp of Keras and
CNNs more generally.
"""

### IMPORT ###

import random
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from PIL import Image

import ai_data_utils

### CONST ###
TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 360
dataset_folder = 'dataset_imgs/'

col_names = []
for num in range(9):
    col_name = 'square_'+str(num)
    col_names.append(col_name)

# Load Data.
data = ai_data_utils.dataset_to_dataframe()

######## FUNCTIONS/CLASSES ########
class RubiksCubeFaceDataGen():
    '''
    Data generator for the images in dataset_imgs. Used to train the model.
    '''

    def __init__(self, df):
        self.df = df

    def generate_indices(self):
        '''
        Shuffle df. Then, create the indices for a training set, test set, and validation set.
        '''
        # Shuffle.
        p = np.random.permutation(len(self.df))

        # Set indices created.
        train_up_to = int(len(self.df)*TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to*TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

        return train_idx, valid_idx, test_idx
    
    def preprocess_and_augment(self, img_path):
        '''
        Preprocesses an image in a batch and performs basic data augmentation using the filter functions from ai_basics.

        Filtering Algorithm:

        Choose a number of filters to apply randomly from 0 to 5.

        For each filter, apply it with a strength from 2%-15% in either a positive or negative direction.
        (Certain filters have limits described in their own documentation and are automatically handled.)

        Normalize the image and return it.
        '''

        # Open the image, convert it to an array.
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.asarray(im)

        num_filters = random.randint(0, 5)
        if num_filters == 0:
            pass
        else:
            for x in range(num_filters):
                positive = bool(random.randint(0,1))

                if positive:
                    amount = random.randint(2, 15)
                else:
                    amount = -1*random.randint(2, 15)
                
                filterfunc = random.choice([ai_data_utils.redden,
                                             ai_data_utils.greenify,
                                             ai_data_utils.blueify,
                                             ai_data_utils.saturate,
                                             ai_data_utils.change_temp,
                                             ai_data_utils.brighten,
                                             ai_data_utils.contrast,
                                             ai_data_utils.blur])
                
                im = filterfunc(im, amount)

        im = np.array(im) / 255.0

        return im
    
    def image_batch(self, image_idx, in_training, batchsize=10):
        '''
        Generates image batches when using the model.
        '''

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

                im = self.preprocess_and_augment(file)

                sizes.append(size)
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

class RubiksOutputModel():
    """
    Generates the multi-output model with 11 branches. Each has a sequence of layers defined by make_default_hidden_layers.
    """
    def make_default_hidden_layers(self, inputs):
        """
        Generate the default set of hidden layers.

        Conv2D-> BatchNorm -> Pooling -> Dropout.
        """
        x = Conv2D(16, (3,3), padding='same')(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3,3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3,3), padding='same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3,3), padding='same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        return x
    
    def build_sizes(self, inputs):
        """
        Builds the size branch.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("sigmoid", name="size_out")(x)

        return x
    
    def build_edges(self, inputs):
        """
        Builds the edgedness branch -- that is, is the cube edged or edgeless.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("sigmoid", name="edged_out")(x)

        return x
    
    def build_square0(self, inputs):
        """
        Builds the branch for square 0.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq0out")(x)

        return x
    
    def build_square1(self, inputs):
        """
        Builds the branch for square 1.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq1out")(x)

        return x
    
    def build_square2(self, inputs):
        """
        Builds the branch for square 2.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq2out")(x)

        return x
    
    def build_square3(self, inputs):
        """
        Builds the branch for square 3.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq3out")(x)

        return x
    
    def build_square4(self, inputs):
        """
        Builds the branch for square 4.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq4out")(x)

        return x
    
    def build_square5(self, inputs):
        """
        Builds the branch for square 5.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq5out")(x)

        return x
    
    def build_square6(self, inputs):
        """
        Builds the branch for square 6.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq6out")(x)

        return x
    
    def build_square7(self, inputs):
        """
        Builds the branch for square 7.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq7out")(x)

        return x
    
    def build_square8(self, inputs):
        """
        Builds the branch for square 8.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(6)(x)
        x = Activation("softmax", name="sq8out")(x)

        return x
    
    def assemble_cnn(self, width, height):
        '''
        Used to assemble the full multioutput model.
        '''

        input_shape = (height, width, 3)

        inputs = Input(shape=input_shape)

        sizebr = self.build_sizes(inputs)
        edgedbr = self.build_edges(inputs)
        sq0br = self.build_square0(inputs)
        sq1br = self.build_square1(inputs)
        sq2br = self.build_square2(inputs)
        sq3br = self.build_square3(inputs)
        sq4br = self.build_square4(inputs)
        sq5br = self.build_square5(inputs)
        sq6br = self.build_square6(inputs)
        sq7br = self.build_square7(inputs)
        sq8br = self.build_square8(inputs)

        model = Model(inputs=inputs, outputs=[sizebr, edgedbr, sq0br, sq1br, sq2br, sq3br, sq4br, sq5br, sq6br, sq7br, sq8br], name="rubiks_net")

        return model

############ MAIN ############
def main():
    ''' Train the AI with preset hyperparameters. Checkpoints and weights are saved in the model_checkpoint folder.'''

    # Generate data and indices
    datagen = RubiksCubeFaceDataGen(data)
    train_idx, valid_idx, test_idx = datagen.generate_indices()
    model = RubiksOutputModel().assemble_cnn(IM_WIDTH, IM_HEIGHT)

    # Learning Rate, number of Epochs
    init_lr = 1e-4
    epochs = 150


    # Model Compilation and Weights
    opt = Adam(learning_rate=init_lr, decay=init_lr/epochs)

    model.compile(optimizer=opt,loss={'size_out':'binary_crossentropy',
                                    'edged_out':'binary_crossentropy',
                                    'sq0out':'categorical_crossentropy',
                                    'sq1out':'categorical_crossentropy',
                                    'sq2out':'categorical_crossentropy',
                                    'sq3out':'categorical_crossentropy',
                                    'sq4out':'categorical_crossentropy',
                                    'sq5out':'categorical_crossentropy',
                                    'sq6out':'categorical_crossentropy',
                                    'sq7out':'categorical_crossentropy',
                                    'sq8out':'categorical_crossentropy'},
                                    loss_weights={
                                        'size_out':6,
                                        'edged_out':6,
                                        'sq0out':2,
                                        'sq1out':2,
                                        'sq2out':2,
                                        'sq3out':2,
                                        'sq4out':2,
                                        'sq5out':2,
                                        'sq6out':2,
                                        'sq7out':2,
                                        'sq8out':2},
                                        metrics={
                                            'size_out':'accuracy',
                                            'edged_out':'accuracy',
                                            'sq0out':'accuracy',
                                            'sq1out':'accuracy',
                                            'sq2out':'accuracy',
                                            'sq3out':'accuracy',
                                            'sq4out':'accuracy',
                                            'sq5out':'accuracy',
                                            'sq6out':'accuracy',
                                            'sq7out':'accuracy',
                                            'sq8out':'accuracy'})


    # Batch Sizes.
    batch_size=32
    valid_batch_size=32

    # Generate image batches based on the training and validation indices.
    train_gen = datagen.image_batch(train_idx, in_training=True, batchsize=batch_size)
    valid_gen = datagen.image_batch(valid_idx, in_training=True, batchsize=valid_batch_size)

    callbacks = [ModelCheckpoint("./model_checkpoint", monitor = 'val_loss')]

    # Train Model
    history = model.fit(train_gen, steps_per_epoch=len(train_idx)//batch_size, epochs=epochs, callbacks=callbacks, validation_data=valid_gen, validation_steps = len(valid_idx)//valid_batch_size)