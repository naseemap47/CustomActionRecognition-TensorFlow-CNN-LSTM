import random
import numpy as np
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import argparse
import time
import mlflow

from keras.callbacks import EarlyStopping
from models import convlstm_model, LRCN_model
from keras.preprocessing.image import ImageDataGenerator
from utils import VideoFrameGenerator


seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to csv Data")
# Specify the number of frames of a video that will be fed to the model as one sequence.
ap.add_argument("-l", "--seq_len", type=int, default=20,
                help="length of Sequence")

ap.add_argument("-s", "--size", type=int, default=64,
                help="Specify the height and width to which each video frame will be resized in our dataset.")
ap.add_argument("-m", "--model", type=str,  default='LRCN',
                choices=['convLSTM', 'LRCN'],
                help="select model type convLSTM or LRCN")
ap.add_argument("-e", "--epochs", type=int, default=70,
                help="number of epochs")
ap.add_argument("-b", "--batch_size", type=int, default=4,
                help="number of batch_size")

args = vars(ap.parse_args())
DATASET_DIR = args["dataset"]
SEQUENCE_LENGTH = args["seq_len"]
IMAGE_SIZE = args["size"]
model_type = args["model"]
epochs = args["epochs"]
batch_size = args["batch_size"]

# some global params
SIZE = (IMAGE_SIZE, IMAGE_SIZE)
CHANNELS = 3

# pattern to get videos and classes
glob_pattern= DATASET_DIR + '/{classname}/*'

# Data Extraction Start
s_time = time.time()

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = sorted(os.listdir(DATASET_DIR))

# for data augmentation
# preprocessor = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1
# )

# Create video frame generator
train_gen = VideoFrameGenerator(
    classes=CLASSES_LIST, 
    glob_pattern=glob_pattern,
    nb_frames=SEQUENCE_LENGTH,
    split=.1, 
    shuffle=True,
    batch_size=batch_size,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    # transformation=preprocessor,
    use_frame_cache=False
)

# Validation Generator
valid_gen = train_gen.get_validation_generator()

# Data Size
train_size = int(train_gen.files_count)
val_size = int(valid_gen.files_count)
total_data = train_size + val_size

# Model Selection
if model_type == 'convLSTM':
    print("\33[5;30;43m [INFO] Selected convLSTM Model \33[0m")
    model = convlstm_model(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST)
    print("\33[1;37;42m [INFO] convLSTM Created Successfully \33[0m")
elif model_type == 'LRCN':
    print("\33[5;30;43m [INFO] Selected LRCN Model \33[0m")
    model = LRCN_model(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST)
    print("\33[1;37;42m [INFO] LRCN Created Successfully \33[0m")
else:
    print('\33[91m [INFO] Model NOT Choosen!! \33[0m')

# Model Dir
path_to_model_dir = f'{model_type}'
if not os.path.isdir(path_to_model_dir):
    os.makedirs(path_to_model_dir, exist_ok=True)
    print(f'\33[92m [INFO] Created {path_to_model_dir} Folder \33[0m')
else:
    print(f'\33[93m [INFO] {path_to_model_dir} Folder Already Exist \33[0m')
    f = glob.glob(path_to_model_dir + '/*')
    for i in f:
        os.remove(i)

png_name = f'{model_type}_model_str.png'
path_to_model_str = os.path.join(path_to_model_dir, png_name)
# Plot the structure of the contructed model.
tf.keras.utils.plot_model(model, to_file=path_to_model_str,
           show_shapes=True, show_layer_names=True)
print(f'\33[92m [INFO] Successfully Created {png_name} \33[0m')

# Create an Instance of Early Stopping Callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

# Compile the model and specify loss function, optimizer and metrics values to the model
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=["accuracy"])

print(f'\33[1;37;44m [INFO] {model_type} Model Training Started... \33[0m')

# MLFlow
mlflow.set_experiment('Action Recognition')
with mlflow.start_run(run_name=f'{model_type}_model'):
    mlflow.tensorflow.autolog()
    # Start training the model.
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping_callback]
    )

    print(f'\33[1;37;42m [INFO] Successfully Completed {model_type} Model Training \33[0m')

    # Training End
    te_time = time.time()
    t2 = (te_time-s_time)/60
    print(f'\33[5;30;46m [INFO] Model Training Completed in {round(t2, 2)} Minutes \33[0m')
    
    # Evaluate the trained model.
    model_evaluation_history = model.evaluate(valid_gen)

    # Get the loss and accuracy from model_evaluation_history.
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

    # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
    model_file_name = f'{model_type}_model_loss_{model_evaluation_loss:.3}_acc_{model_evaluation_accuracy:.3}.h5'

    # Save your Model
    path_to_save_model = os.path.join(path_to_model_dir, model_file_name)
    model.save(path_to_save_model)
    print(f'\33[1;37;42m [INFO] Model {model_file_name} saved Successfully.. \33[0m')

    # Model Size
    mb_size = os.path.getsize(f'{path_to_save_model}')
    mb_size = round(mb_size / 1e+6, 2)
    print(f'\33[5;30;47m [INFO] {model_type} Model Size: {mb_size} MB \33[0m')

    # Plot History
    metric_loss = history.history['loss']
    metric_val_loss = history.history['val_loss']
    metric_accuracy = history.history['accuracy']
    metric_val_accuracy = history.history['val_accuracy']

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_loss))

    # Plot the Graph.
    plt.plot(epochs, metric_loss, 'blue', label=metric_loss)
    plt.plot(epochs, metric_val_loss, 'red', label=metric_val_loss)
    plt.plot(epochs, metric_accuracy, 'magenta', label=metric_accuracy)
    plt.plot(epochs, metric_val_accuracy, 'green', label=metric_val_accuracy)

    # Add title to the plot.
    plt.title(str('Model Metrics'))

    # Add legend to the plot.
    plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])

    # If the plot already exist, remove
    metrics_png_name = f'{model_type}_metrics.png'
    path_to_metrics = os.path.join(path_to_model_dir, metrics_png_name)
    plot_png = os.path.exists(path_to_metrics)
    if plot_png:
        os.remove(path_to_metrics)
        plt.savefig(path_to_metrics, bbox_inches='tight')
    else:
        plt.savefig(path_to_metrics, bbox_inches='tight')
    print(f'\33[1;37;42m [INFO] Successfully Saved {metrics_png_name} \33[0m')

    # MLFlow Metrics
    mlflow.log_metric('Input Image Size', IMAGE_SIZE)
    mlflow.log_metric('Total Image Data', total_data)
    mlflow.log_metric('Train Size', train_size)
    mlflow.log_metric('Validation Size', val_size)
    mlflow.log_artifact(f'{path_to_metrics}')
    mlflow.log_metric('Model Size MB', mb_size)

    print("\33[1;30;43m [INFO] MLFlow Run: \33[0m", mlflow.active_run().info.run_uuid)
mlflow.end_run()

# Total Time
e_time = time.time()
t3 = (e_time-s_time)/60
print(f'\33[5;30;46m [INFO] Completed All process in {round(t3, 2)} Minutes \33[0m')
