import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def load_and_preprocess_image(filename: str = 'Calico.jpg', path: str = './images/'):
    file_path = os.path.join(path, filename)
    img = keras.preprocessing.image.load_img(file_path, target_size=(180, 180))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array


def load_model(num_classes: int = 4, path_to_weights: str = './models/final/cp.ckpt'):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255,
                                                    input_shape=(180, 180, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.load_weights(path_to_weights)
    return model


def get_prediction(image, model):
    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])
    class_names = ['Calico', 'Tabby', 'Tortoiseshell', 'Tuxedo']
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return class_names, np.array(score)


if __name__ == '__main__':
    model = load_model()
    image = load_and_preprocess_image()
    class_names, score = get_prediction(image, model)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
