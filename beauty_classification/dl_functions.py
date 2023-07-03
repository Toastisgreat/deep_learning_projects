import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from numpy import array
from tqdm import tqdm

'''
TODO:
    Finish evaluation
    optimize
    Bayesian Neural Network
'''
def load_data(filename: str, file_col: int, value_col: int, ttv_col: int, ttv_value: str, value_dict: dict):

    dataframe = pd.read_csv(filename)

    col_names = dataframe.columns.values

    values = dataframe[dataframe[col_names[ttv_col]] == ttv_value]
    

    X = values[col_names[file_col]]
    y = values[col_names[value_col]]

    #replace 'average'/'beautiful' with a 1 or a 0 for y values
    y = [[value_dict[i]] for i in y]
    
    #shuffle values for better accuracy
    X, y = shuffle(X, y, random_state=25)

    y = tf.convert_to_tensor(y)
    return X, y

def read_images(folder: str, image_list: list, image_size: tuple) -> list:

    final = []
    for image_path in image_list:
        try:
            #open image file
            image = cv2.imread(folder + image_path)

            #change color and resize image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, image_size)

            #convert image to numpy array for training
            final.append(array(image, dtype='float32'))
        except:
            print(image_path)

    return array(final)


def conv_nn(X_test: array, X_valid: array, y_test: tf.Tensor, y_valid: tf.Tensor, scale: int) -> tf.keras.Sequential():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(scale, activation='softmax'))

    model.summary()
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history  = model.fit(
        X_test, y_test,
        epochs=10,
        validation_data=(X_valid, y_valid))
    
    print(f'Convolutional Neural Network\nTest accuracy: {round(model.evaluate(X_test, y_test, verbose=0)[1], 2)}\nValid accuracy: {round(model.evaluate(X_valid, y_valid, verbose=0)[1], 2)}')
    return model, history

def plot_accuracy_loss(history: dict) -> None:

    fig = plt.figure(figsize=(10,5))

    plt.subplot(221)
    plt.plot(history['accuracy'], 'bo--', label= 'acc')
    plt.plot(history['val_accuracy'], 'ro--', label= 'val_acc')
    plt.title('Train acc vs Valid acc')
    plt.ylabel('acc')
    plt.xlabel('epochs')

    #plot loss fn
    plt.subplot(222)
    plt.plot(history['loss'], 'bo--', label='acc')
    plt.plot(history['val_loss'], 'ro--', label='val_loss')
    plt.title('train loss vs valid loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')

    plt.legend()
    plt.show()