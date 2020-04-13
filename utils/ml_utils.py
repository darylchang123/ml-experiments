import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import pickle
import time

import urllib3
urllib3.disable_warnings()

SHUFFLE_SEED = 524287
SHUFFLE_BUFFER_SIZE = 1000
IMG_SIZE = 128


class ModelState():
    def __init__(self, weights, history, times):
        self.weights=weights
        self.history=history
        self.times=times
    weights = None
    history = None 
    times = None


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def load_dataset(dataset_name, shuffle_seed=SHUFFLE_SEED):
    """
    Loads the tensorflow datasets
    :param dataset_name: One of the following dataset names: https://www.tensorflow.org/datasets/catalog/overview
    :param shuffle_seed: Seed for shuffling
    :return: (raw_train, raw_validation), label_metadata
    """
    read_config = tfds.ReadConfig(shuffle_seed=shuffle_seed)
    if dataset_name == 'cats_vs_dogs':
        (raw_train, raw_validation), metadata = tfds.load(
            'cats_vs_dogs',
            split=['train[:80%]', 'train[80%:]'],
            shuffle_files=True,
            with_info=True,
            as_supervised=True,
            read_config=read_config
        )
        print('Training Data Summary')
        summarize_dataset(raw_train)
        print('\nValidation Data Summary')
        summarize_dataset(raw_validation)
        return (raw_train, raw_validation), metadata.features['label'].names
    return None


def summarize_dataset(tf_data):
    """
    Prints stats around no. of classes in the dataset
    :param tf_data: PrefetchDataset
    """
    label = np.array([l for _, l in tf_data])
    class_freq = np.array(np.unique(label, return_counts=True)).transpose()
    class_summary = {f[0]: (f[1], f[1] * 100 / len(label)) for f in class_freq}
    print('No. of examples: {count}'.format(count=len(label)))
    for k,v in class_summary.items():
        print('Class: {class_val} :::: Count: {count} :::: Percentage: {percent}'.format(
            class_val=k,
            count=v[0],
            percent=v[1]
        ))


def resize_image(image, label, img_size):
    """
    Resizes image
    :param image: Image
    :param label: Label
    :param img_size: Image size
    :return:
    """
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (img_size, img_size))
    return image, label


def show_image(image, label, label_names):
    """
    Shows image
    :param image: Image
    :param label: Label
    :param label_names: List containing label names
    :return:
    """
    plt.figure()
    plt.imshow(image)
    plt.title('Class: {class_value} :::: Class Name: {label_name}'.format(
        class_value=label,
        label_name=label_names[label] if len(label_names)>label else ''
    ))


def load_batched_and_resized_dataset(
    dataset_name,
    batch_size,
    img_size=IMG_SIZE,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    shuffle_seed=SHUFFLE_SEED
):
    """
    Resizes and normalizes images, caches them in memory, and divides them into batches
    :param dataset_name: One of the following dataset names: https://www.tensorflow.org/datasets/catalog/overview
    :param batch_size: Batch size
    :param img_size: Target image size, defaults to IMG_SIZE
    :param shuffle_buffer_size: Number of examples to load into buffer for shuffling, defaults to SHUFFLE_BUFFER_SIZE
    :param shuffle_seed: Seed for shuffling, defaults to SHUFFLE_SEED
    :return: train_batches, validation_batches
    """
    # Load dataset
    (raw_train, raw_validation), label_names = load_dataset('cats_vs_dogs', shuffle_seed=shuffle_seed)
    
    # Resize images and normalize (divide by 255)
    train = raw_train.map(lambda img, lbl: resize_image(img, lbl, img_size))
    validation = raw_validation.map(lambda img, lbl: resize_image(img, lbl, img_size))
    
    # Cache data in memory
    train = train.cache()
    validation = validation.cache()
    
    # Divide data into batches
    train_batches = train.shuffle(
        buffer_size=shuffle_buffer_size,
        seed=shuffle_seed,
        reshuffle_each_iteration=False,
    ).batch(batch_size)
    validation_batches = validation.batch(batch_size)
    
    return train_batches, validation_batches


def build_model(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    initializer=keras.initializers.glorot_uniform(seed=0),
):
    """
    Builds a base model according to the parameters specified. Architecture is similar to VGG16.
    :param optimizer: Type of optimizer to use, along with corresponding optimizer settings
    :param initializer: Kernel initializer to use for each layer
    :return: Compiled Keras model
    """
    model = keras.models.Sequential([
        layers.Conv2D(
            input_shape=(IMG_SIZE,IMG_SIZE,3),
            filters=4,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=initializer,
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(
            filters=8,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=initializer,
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=initializer,
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=initializer,
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=initializer,
        ),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_initializer=initializer),
        layers.Dense(256, activation='relu', kernel_initializer=initializer),
        layers.Dense(1, activation='sigmoid', kernel_initializer=initializer),
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train, validation, epochs):
    time_callback = TimeHistory()
    history = model.fit(
        train,
        epochs=epochs,
        validation_data=validation,
        callbacks=[time_callback]
    )
    return get_model_state(model, history, time_callback)


def summarize_diagnostics(history):
    """
    # Plot diagnostic learning curves
    :param history: Keras history object
    :return: Show plots
    """
    # Plot loss
    plt.figure(figsize=(10, 10), dpi=80)
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history['loss'], color='blue', label='train')
    plt.plot(history['val_loss'], color='orange', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.legend()
    # Plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history['accuracy'], color='blue', label='train')
    plt.plot(history['val_accuracy'], color='orange', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_accuracies_by_param(param_values, history_dict, param_name, filename):
    """
    Given a set of parameter values (e.g. batch sizes) and histories, this function
    creates two plots: one of training accuracy and another of validation accuracy
    :param param_values: List of parameter values used to generate histories (e.g. batch sizes)
    :param history_dict: Dictionary from param value to a Keras history.history
    :param param_name: String name of the parameter (e.g. 'batch size')
    :param filename: file to save the plot to
    """
    plt.figure(figsize=(10, 10), dpi=80)
    plt.subplot(211)
    plt.title('Effect of {} on training accuracy'.format(param_name))
    for param_value in param_values:
        if param_value in history_dict:
            plt.plot(history_dict[param_value]['accuracy'], label=str(param_value))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend(loc='best')
    
    plt.subplot(212)
    plt.title('Effect of {} on validation accuracy'.format(param_name))
    for param_value in param_values:
        if param_value in history_dict:
            plt.plot(history_dict[param_value]['val_accuracy'], label=str(param_value))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend(loc='best')

    plt.show()
    plt.savefig('graphs/{}'.format(filename))


def plot_loss_by_param(model_state_by_type, param_name, filename):
    """
    Given a set of parameter values (e.g. batch sizes) and histories, this function
    creates two plots: one of training accuracy and another of validation accuracy
    :param param_values: List of parameter values used to generate histories (e.g. batch sizes)
    :param history_dict: Dictionary from param value to a Keras history.history
    :param param_name: String name of the parameter (e.g. 'batch size')
    :param filename: file to save the plot to
    """
    plt.figure(figsize=(10, 10), dpi=80)
    plt.subplot(211)
    plt.title('Effect of {} on training loss'.format(param_name))
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['loss'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(loc='best')
    
    plt.subplot(212)
    plt.title('Effect of {} on validation loss'.format(param_name))
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['val_loss'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(loc='best')

    plt.show()
    plt.savefig('graphs/{}'.format(filename))
    

def get_model_state(model, model_history, time_callback):
    model_state = ModelState()
    model_state.history = model_history.history
    model_state.times = time_callback.times
    model_state.weights = [w.value() for w in model.weights]
    return model_state
    
    
def save_model_state(model_state, filename):
    pickle.dump(model_state, open("pickled_objects/{filename}.pickle".format(filename=filename), "wb" ))
    
    
def load_model_state(filename):
    return pickle.load(open("pickled_objects/{filename}.pickle".format(filename=filename), "rb" ))

