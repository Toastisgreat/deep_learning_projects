import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras

import cv2

import matplotlib.pyplot as plt
from pathlib import PurePath

'''TODO:
    - Tensorflow 2.10 fix
    - consider adding extra datasets to this concept?

'''
def read_images(folder: str, image_size: tuple) -> list:
    image_list = []
    for path,subdirs,files in os.walk(folder):
            for name in files:
                pp = PurePath(path, name)
                
                if pp.suffix in ['.jpeg','.jpg','.png']:
                    image_list.append([pp,pp.parent])

    final = []
    final_vals = []
    threed_final = []
    for image_path_duo in image_list:
        image_path = image_path_duo[0]
        try:
            #open image file
            image = cv2.imread(image_path.as_posix())

            #change color and resize image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, image_size)

            threed_final.append(np.dstack((image,image,image)))
            #convert image to numpy array for training
            final.append(np.array(image, dtype='float32'))
            final_vals.append(image_path_duo[1])
        except Exception as e:
            print(f'{image_path} failed to render, make sure its in a correct format. check docs for formats\n{e}')

    return np.array(final), final_vals, np.array(threed_final)

def convert_to_integer(final_vals:list, conversion_dict):
    final_converted = []
    for path in final_vals:
        try:
            final_converted.append(conversion_dict[path.stem])
        except:
            print(f'Stem is not in dict: {path.stem}')
    return final_converted

class data_augmentation_layer(keras.Sequential):
    def __init__(self, image_size):
        super(data_augmentation_layer, self).__init__()
        
        #create outline for the layers
        self.Normlayer = keras.layers.Normalization()
        self.Resizinglayer = keras.layers.Resizing(image_size,image_size)
        self.rand_flip = keras.layers.RandomFlip()
        self.rand_rotate = keras.layers.RandomRotation(factor=0.02)
        self.rand_zoom = keras.layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        )

    def call(self, inputs):

        #creat the model
        model = self.Normlayer(inputs)
        model = self.Resizinglayer(model)
        model = self.rand_flip(model)
        model = self.rand_rotate(model)
        model = self.rand_zoom(model)

        return model

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=tf.nn.sigmoid)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
def make_data_augmentation_layer(image_size):
    model = keras.Sequential([
            keras.layers.Normalization(),
            keras.layers.Resizing(image_size,image_size),
            keras.layers.RandomFlip(),
            keras.layers.RandomRotation(factor=0.02),
            keras.layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),],
            name="data_augmentation_layer"
            )

    return model

def vit_classifier(input_shape,patch_size, num_patches, projection_dim,transformer_layers, num_heads,transformer_units, mlp_head_units, image_size, num_classes):

    '''
    Parameter details:
    input_shape = (x,y,z) :: shape of each image from read_images 3d thing
    patch_size = int :: size of patches for each image (patch, patch)
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64 :: vector size for patch encoder
    transformer_layers = int :: how many transofrmer layers u want, supposed to be hyperparam
    num_heads = int :: number of heads in multi layer attention layer
    transformer_units = = [
        projection_dim * 2,
        projection_dim,
        ]  # Size of the transformer layers
    
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
    '''

    #input layer for shaping
    inputs = keras.layers.Input(shape=input_shape)

    #Augment data
    data_aug = make_data_augmentation_layer(image_size)
    data_aug = data_aug(inputs)

    #patches
    patches = Patches(patch_size)(data_aug)

    #patch encoder
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    #Create multiple ViT layers
    for layerno in range(transformer_layers):
        
        # Layer normalization 1.
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = keras.layers.Add()([x3, x2])

    
    # Create a [batch_size, projection_dim] tensor.
    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.

    logits = keras.layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)

    return model

def compile_and_test(model, batch_size, num_epochs, x_train,y_train,x_test,y_test):

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history

