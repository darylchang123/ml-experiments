import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from functools import reduce
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
import os
import pickle
import random
import scipy
import time



############################ INITIALIZATION FUNCTIONS #########################

def init_env():
    """
    Sets environment variables and seeds to make model training deterministic
    """
    # Make GPU use deterministic algorithms (see https://github.com/NVIDIA/tensorflow-determinism)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Seed value (can actually be different for each attribution step)
    seed_value = 0

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)


############################ DATASET FUNCTIONS ################################

def load_dataset(dataset_name, shuffle_seed):
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


def resize_image(image, label, img_size, normalize_pixel_values=True):
    """
    Resizes image
    :param image: Image
    :param label: Label
    :param img_size: Image size
    :param normalize_pixel_values: Whether to divide pixel values by 255
    :return:
    """
    image = tf.cast(image, tf.float32)
    if normalize_pixel_values:
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
    batch_size=32,
    img_size=128,
    shuffle_buffer_size=1000,
    shuffle_seed=0,
    normalize_pixel_values=True
):
    """
    Resizes and normalizes images, caches them in memory, and divides them into batches
    :param dataset_name: One of the following dataset names: https://www.tensorflow.org/datasets/catalog/overview
    :param batch_size: Batch size
    :param img_size: Target image size, defaults to 128
    :param shuffle_buffer_size: Number of examples to load into buffer for shuffling, defaults to 1000
    :param shuffle_seed: Seed for shuffling, defaults to 0
    :param normalize_pixel_values: Whether to divide pixel values by 255.
    :return: train_batches, validation_batches
    """
    # Load dataset
    (raw_train, raw_validation), label_names = load_dataset(dataset_name, shuffle_seed=shuffle_seed)
    
    # Resize images and normalize (divide by 255) if specified
    resize = lambda img, lbl: resize_image(img, lbl, img_size, normalize_pixel_values)
    train = raw_train.map(resize)
    validation = raw_validation.map(resize)
    
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


############################ MODEL BUILDING AND TRAINING FUNCTIONS ################################
    
class ModelState():
    def __init__(
        self, 
        weights=None, 
        history=None, 
        times=None,
    ):
        self.weights = weights
        self.history = history
        self.times = times


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

        
def build_model(
    conv_dropout_rate=0,
    dense_dropout_rate=0.7,
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    initializer=keras.initializers.glorot_uniform(seed=0),
    seed_value=0,
    conv_l1_regularizer=0,
    conv_l2_regularizer=0,
    dense_l1_regularizer=0,
    dense_l2_regularizer=0,
    input_shape=(128,128,3),
    use_batch_normalization=True,
):
    """
    Builds a base model according to the parameters specified. Architecture is similar to VGG16.
    :param dropout_rate: Dropout rate to use
    :param optimizer: Type of optimizer to use, along with corresponding optimizer settings
    :param initializer: Kernel initializer to use for each layer
    :param seed_value: Seed value to use for dropout layer
    :param l1_regularizer: L1 regularizer instance
    :param l2_regularizer: L2 regularizer instance
    :param input_shape: Shape of image input
    :return: Compiled Keras model
    """
    def add_batch_norm(is_input=False):
        if use_batch_normalization:
            if is_input:
                model.add(layers.BatchNormalization(input_shape=input_shape))
            else:
                model.add(layers.BatchNormalization())
            
    model = keras.models.Sequential()
    add_batch_norm(is_input=True)
    if use_batch_normalization:
        model.add(layers.Conv2D(
            filters=4,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer),
            bias_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer)
        ))
    else:
        model.add(layers.Conv2D(
            input_shape=input_shape,
            filters=4,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer),
            bias_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer)
        ))
    add_batch_norm()
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(
        rate=conv_dropout_rate,
        seed=seed_value
    ))
    model.add(layers.Conv2D(
        filters=8,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer),
        bias_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer)
    ))
    add_batch_norm()
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(
        rate=conv_dropout_rate,
        seed=seed_value
    ))
    model.add(layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer),
        bias_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer)
    ))
    add_batch_norm()
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(
        rate=conv_dropout_rate,
        seed=seed_value
    ))
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer),
        bias_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer)
    ))
    add_batch_norm()
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(
        rate=conv_dropout_rate,
        seed=seed_value
    ))
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer),
        bias_regularizer=regularizers.l1_l2(l1=conv_l1_regularizer, l2=conv_l2_regularizer)
    ))
    add_batch_norm()
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(
        rate=dense_dropout_rate,
        seed=seed_value
    ))
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=256, 
        activation=None, 
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l1_l2(l1=dense_l1_regularizer, l2=dense_l2_regularizer),
        bias_regularizer=regularizers.l1_l2(l1=dense_l1_regularizer, l2=dense_l2_regularizer)
    ))
    add_batch_norm()
    model.add(layers.ReLU())
    model.add(layers.Dropout(
        rate=dense_dropout_rate,
        seed=seed_value
    ))
    model.add(layers.Dense(
        units=256, 
        activation=None, 
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l1_l2(l1=dense_l1_regularizer, l2=dense_l2_regularizer),
        bias_regularizer=regularizers.l1_l2(l1=dense_l1_regularizer, l2=dense_l2_regularizer)
    ))
    add_batch_norm()
    model.add(layers.ReLU())
    model.add(layers.Dropout(
        rate=dense_dropout_rate,
        seed=seed_value
    ))
    model.add(layers.Dense(
        units=1, 
        activation='sigmoid', 
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l1_l2(l1=dense_l1_regularizer, l2=dense_l2_regularizer),
        bias_regularizer=regularizers.l1_l2(l1=dense_l1_regularizer, l2=dense_l2_regularizer)
    ))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model



def train_model(model, train, validation, epochs, extra_callbacks=[]):
    time_callback = TimeHistory()
    history = model.fit(
        train,
        epochs=epochs,
        validation_data=validation,
        callbacks=[time_callback] + extra_callbacks
    )
    return get_model_state(model, history, time_callback)
    

def get_model_state(model, model_history, time_callback):
    model_state = ModelState()
    model_state.history = model_history.history
    model_state.times = time_callback.times
    model_state.weights = [w.value() for w in model.weights]
    return model_state
    
    
def save_model_state(model_state, filename):
    model_state_serialize={}
    for key,state in model_state.items():
        model_state_serialize[key]=(state.weights,state.history,state.times)
    pickle.dump(model_state_serialize, 
                open("pickled_objects/{filename}.pickle".format(filename=filename), "wb" ))
    
    
def load_model_state(filename):
    model_state_serialize=pickle.load(open("pickled_objects/{filename}.pickle".format(filename=filename), "rb" ))
    model_state_by_key={}
    for key,state in model_state_serialize.items():
        model_state_by_key[key]=ModelState(weights=state[0],history=state[1],times=state[2])
    return model_state_by_key

########################################## VISUALIZATIONS AND METRICS #################################################
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
    

def plot_accuracies_by_param(model_state_by_type, param_name, filename, ylim_left=None, ylim_right=None):
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
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['accuracy'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(ylim_left,ylim_right)
        plt.grid(True)
        plt.legend(loc='best')
    
    plt.subplot(212)
    plt.title('Effect of {} on validation accuracy'.format(param_name))
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['val_accuracy'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(ylim_left,ylim_right)
        plt.grid(True)
        plt.legend(loc='best')

    plt.show()
    plt.savefig('graphs/{}'.format(filename))


def plot_loss_by_param(model_state_by_type, param_name, filename, ylim_left=None, ylim_right=None):
    """
    Given a set of parameter values (e.g. batch sizes) and histories, this function
    creates two plots: one of training loss and another of validation loss
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
        plt.ylim(ylim_left,ylim_right)
        plt.grid(True)
        plt.legend(loc='best')
    
    plt.subplot(212)
    plt.title('Effect of {} on validation loss'.format(param_name))
    for typ, state in model_state_by_type.items():
        plt.plot(state.history['val_loss'], label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(ylim_left,ylim_right)
        plt.grid(True)
        plt.legend(loc='best')

    plt.show()
    plt.savefig('graphs/{}'.format(filename))
    

def plot_generalization_gap_by_param(model_state_by_type, param_name, filename, clipping_val=None, ylim_left=None, ylim_right=None):
    """
    Given a set of parameter values (e.g. batch sizes) and histories, this function
    creates one plot representing generalization gap: val_loss/train_loss
    :param param_values: List of parameter values used to generate histories (e.g. batch sizes)
    :param history_dict: Dictionary from param value to a Keras history.history
    :param param_name: String name of the parameter (e.g. 'batch size')
    :param filename: file to save the plot to
    """
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Effect of {} on generalization gap'.format(param_name))
    for typ, state in model_state_by_type.items():
        gen_gap=np.array(state.history['val_loss'])/np.array(state.history['loss'])
        if clipping_val:
            gen_gap=np.clip(gen_gap, None, clipping_val)
        plt.plot(gen_gap, label=str(typ))
        plt.xlabel('Epoch')
        plt.ylabel('Genralization Gap')
        plt.ylim(ylim_left,ylim_right)
        plt.grid(True)
        plt.legend(loc='best')
    plt.savefig('graphs/{}'.format(filename))

    
def visualize_weights(weights_by_key, filename, bins=None):
    if not bins:
        bins=[0.005*a-0.3 for a in range(120)]
    plt.figure(figsize=(10, 10), dpi=80)
    plt.title('Distribution of weights by model')

    for model, weight in weights_by_key.items():
        flat_weight=np.ndarray.flatten(weight)
        max_wt=np.max(flat_weight)
        min_wt=np.max(flat_weight)
        
        print('Model: {model}, Max Weight: {max_wt}, Min Weight: {min_wt}'.format(**locals()))
        sns.distplot(flat_weight, label=str(model), kde=False, bins=bins, )
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend(loc='best')
    plt.savefig('graphs/{}'.format(filename))

    
def get_num_elems_from_shape(shape):
    """
    Given the shape of a vector, returns how many elements are in the vector
    :param shape: shape of vector
    :return: number of elements
    """
    return reduce(lambda x,y: x*y, shape)


def unflatten_weights(flattened_weights, weight_shapes):
    """
    Given a flattened vector of weights, and the desired shapes for each layer, this fn reshapes the weights.

    :param flattened_weights: 1-dimensional vector containing weight values
    :param weight_shapes: list of shapes (one per layer in model)
    :return: list containing reshaped weight vectors (one per layer in model)
    """
    if len(flattened_weights) != sum([get_num_elems_from_shape(shape) for shape in weight_shapes]):
        print("Weight shapes do not match number of flattened weights!")
    i = 0
    unflattened_weights = []
    for shape in weight_shapes:
        num_elems = get_num_elems_from_shape(shape)
        reshaped_weights = np.reshape(flattened_weights[i:i+num_elems], shape)
        unflattened_weights.append(reshaped_weights)
        i += num_elems
    return unflattened_weights


def get_negative_loss(flattened_weights, *args):
    """
    This function sets the last layer of the model to the weights provided, then computes the negative loss.
    This is used as a helper function for get_sharpness.
    
    :param flattened_weights: 1-dimensional vector containing weight values
    :param *args: (model, data, weight_shapes)
    :return: negative loss of model evaluated on the data provided
    """
    model, data, weight_shapes = args
    unflattened_weights = unflatten_weights(flattened_weights, weight_shapes)
    model.set_weights(unflattened_weights)
    loss, accuracy = model.evaluate(data)
    return -loss


def get_negative_loss_gradient(flattened_weights, *args):
    """
    Computes the gradient of the negative loss with respect to the model weights.
    
    :param flattened_weights: flattened model weights
    :param *args: (model, data, weight_shapes)
    :return: flattened gradient with respect to weights (1d vector)
    """
    model, data, weight_shapes = args
    unflattened_weights = unflatten_weights(flattened_weights, weight_shapes)
    model.set_weights(unflattened_weights)
    
    batch_gradients = []
    for x, y in data:
        with tf.GradientTape() as tape:
            preds = model(x)
            negative_loss = tf.math.negative(tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y, preds)))

        gradients = [tf.cast(g, tf.float64).numpy() for g in tape.gradient(negative_loss, model.trainable_variables)]
        flattened_gradients = np.concatenate([g.flatten() for g in gradients])
        batch_gradients.append(flattened_gradients)

    return np.sum(batch_gradients, axis=0)


def get_sharpness(model, data, epsilon=1e-2):
    """
    This function computes the sharpness of a minimizer by maximizing the loss in a neighborhood around the minimizer.
    Based on sharpness metric defined in https://arxiv.org/pdf/1609.04836.pdf.
    
    :param model: model, where the weights represent a minimizer of the loss function
    :param data: data to evaluate the model on
    :param epsilon: controls the size of the neighborhood to explore
    :return: sharpness
    """
    # Get original loss
    original_loss, original_accuracy = model.evaluate(data)
    
    # Compute bounds on weights
    weights = model.get_weights()
    weight_shapes = [w.shape for w in weights]
    flattened_weights = np.concatenate([x.flatten()for x in weights])
    delta = epsilon * (np.abs(flattened_weights) + 1)
    lower_bounds = flattened_weights - delta 
    upper_bounds = flattened_weights + delta
    
    # Create copy of model so we don't modify original
    model.save('pickled_objects/sharpness_model_clone.h5')
    model_clone = keras.models.load_model('pickled_objects/sharpness_model_clone.h5')
    os.remove('pickled_objects/sharpness_model_clone.h5')
    
    # Minimize
    x, f, d = scipy.optimize.fmin_l_bfgs_b(
        func=get_negative_loss,
        fprime=get_negative_loss_gradient,
        x0=flattened_weights,
        args=(model_clone, data, weight_shapes),
        bounds=list(zip(lower_bounds, upper_bounds)),
        maxiter=10,
        maxls=1,
        disp=1,
    )
    
    # Compute sharpness
    sharpness = (-f - original_loss) / (1 + original_loss) * 100
    return sharpness


# Based on https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py#L195
def get_random_filter_normalized_direction(weights):
    """
    Given a set of weights for a model, returns a random Gaussian direction.
    Normalize each convolutional filter or each FC neuron to match the corresponding norm in the weights parameter.
    :param weights: model weights
    """
    random_direction = []
    for w in weights:
        num_dimensions = len(w.shape)
                             
        # For biases, set to 0
        if num_dimensions == 1:
            new_w = np.zeros(w.shape)
        
        # For fully-connected layers, generate random vector for each neuron and normalize
        elif num_dimensions == 2:
            new_w = np.random.randn(*w.shape)
            for f in range(w.shape[-1]):
                new_filter = new_w[:, f]
                old_filter = w[:, f]
                new_filter *= np.linalg.norm(old_filter) / np.linalg.norm(new_filter)
            
        # For convolutional layers, generate random vector for each filter and normalize
        elif num_dimensions == 4:
            new_w = np.random.randn(*w.shape)
            for f in range(w.shape[-1]):
                new_filter = new_w[:, :, :, f]
                old_filter = w[:, :, :, f]
                new_filter *= np.linalg.norm(old_filter) / np.linalg.norm(new_filter)
        
        random_direction.append(new_w)
    return random_direction


def plot_loss_visualization_1d(base_model, training_data, validation_data, title=None, output_filename=None):
    """
    Visualizes the minimizer for a model along a random Gaussian filter-normalized direction.
    :param base_model: model to evaluate
    :param training_data: training data, used to generate training loss numbers
    :param validation_data: validation data, used to generates validation loss numbers
    :param title: title for the plot
    :param output_filename: file to save the plot to
    :return: x_values, train_losses, validation_losses
    """
    # Get weights and generate random direction
    weights = base_model.get_weights()
    direction = get_random_filter_normalized_direction(weights)
    
    # Set up new model and plotting variables
    x_values = np.linspace(-1, 1, 20)
    train_losses = []
    validation_losses = []
    new_model = build_model()
    
    # Compute training and validation loss for each linear combination of weight and direction
    for x in x_values:
        print("\nx: ", x)
        
        # Compute and set weights
        new_weights = [w + x * d for w, d in zip(weights, direction)]
        new_model.set_weights(new_weights)
        
        # Evaluate model
        train_loss, train_accuracy = new_model.evaluate(training_data)
        validation_loss, validation_accuracy = new_model.evaluate(validation_data)
        
        # Store losses
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
    
    # Plot results
    plt.plot(x_values, train_losses, linestyle='solid', label='train')
    plt.plot(x_values, validation_losses, linestyle='dashed', label='validation')
    plt.ylabel('Loss')
    plt.xlabel('Alpha')
    plt.legend()
    if title:
        plt.title(title)
    plt.show()
    if output_filename:
        plt.savefig('graphs/{}'.format(output_filename))
    
    return x_values, train_losses, validation_losses


def plot_loss_visualization_2d(base_model, data, mode='all', title=None, output_filename=None, XYZ=None):
    """
    Visualizes the minimizer for a model along two random Gaussian filter-normalized directions.
    :param base_model: model to evaluate
    :param data: data to evaluate the model on, used to generate loss numbers
    :param mode: plotting mode.
       -'filled_contours': generate contours filled in with colors representing levels
       -'contours': generate contours with no fill
       -'surface': generate 3D surface plot
       -'all': generate all of the above
    :param title: title for the plot
    :param output_filename: file to save the plot to
    :param XYZ: tuple of (X, Y, Z) values. If provided, the function will skip loss computation and directly plot the values.
    :return: X, Y, Z
    """
    # Use XYZ parameter for plotting
    if XYZ:
        X, Y, Z = XYZ
    else:
        # Get weights and generate random directions
        weights = base_model.get_weights()
        direction_one = get_random_filter_normalized_direction(weights)
        direction_two = get_random_filter_normalized_direction(weights)

        # Set up new model and plotting variables
        x_values = np.linspace(-1, 1, 5)
        y_values = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.zeros((len(y_values), len(x_values)))
        new_model = build_model()

        # Compute loss for each linear combination of weight and direction
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                # Compute and set weights
                x = x_values[j]
                y = y_values[i]
                print("\n x: {}, y: {}".format(x, y))
                new_weights = [w + x * d1 + y * d2 for w, d1, d2 in zip(weights, direction_one, direction_two)]
                new_model.set_weights(new_weights)

                # Evaluate model
                loss, accuracy = new_model.evaluate(data)

                # Store losses
                Z[i, j] = loss
        
    # Plot results
    if mode == 'filled_contours':
        plt.contourf(X, Y, Z, levels=np.arange(0, 5, 0.25))
        plt.colorbar()
    elif mode == 'contours':
        CS = plt.contour(X, Y, Z, levels=np.arange(0, 5, 0.25))
        plt.clabel(CS, inline=1, fontsize=8)
    elif mode == 'surface':
        ax = plt.axes(projection='3d')
        ax.view_init(60, 35)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    elif mode == 'all':
        fig = plt.figure(figsize=(5, 15))
        # Plot filled contours
        ax1 = fig.add_subplot(3, 1, 1)
        cf = ax1.contourf(X, Y, Z, levels=np.arange(0, 5, 0.25))
        plt.colorbar(cf, ax=ax1)

        # Plot contours
        ax2 = fig.add_subplot(3, 1, 2)
        cs = ax2.contour(X, Y, Z, levels=np.arange(0, 5, 0.25))
        plt.clabel(cs, inline=1, fontsize=8)

        # Plot surface
        ax3 = fig.add_subplot(3, 1, 3, projection='3d')
        ax3.view_init(60, 35)
        ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        
    # Add title and save plot if specified    
    if title:
        plt.title(title)
    plt.show()
    if output_filename:
        plt.savefig('graphs/{}'.format(output_filename))
    
    return X, Y, Z
