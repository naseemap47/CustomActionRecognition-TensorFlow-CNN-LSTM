import random
import numpy as np
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import argparse
import time
import mlflow

from utils.hubconf import custom
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from my_utils import create_dataset
from actModels import convlstm_model, LRCN_model


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
ap.add_argument("-d", "--yolov7_model", type=str, required=True,
                help="path to YOLOv7 detection model")
ap.add_argument("-dc", "--yolov7_conf", type=float, default=0.6,
                help="YOLOv7 detection model confidenece (0<conf<1)")

args = vars(ap.parse_args())
DATASET_DIR = args["dataset"]
SEQUENCE_LENGTH = args["seq_len"]
IMAGE_SIZE = args["size"]
model_type = args["model"]
epochs = args["epochs"]
batch_size = args["batch_size"]
yolov7_model_path = args["yolov7_model"]
yolov7_conf = args["yolov7_conf"]

# YOLOv7 Model
yolov7_model = custom(path_or_model=yolov7_model_path)

# Data Extraction Start
s_time = time.time()

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = sorted(os.listdir(DATASET_DIR))

# Create the dataset.
features, labels, video_files_paths = create_dataset(
    CLASSES_LIST, DATASET_DIR, SEQUENCE_LENGTH, IMAGE_SIZE, yolov7_model, yolov7_conf)

# Total Data Size
total_data = len(labels)

# Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
one_hot_encoded_labels = tf.keras.utils.to_categorical(labels)

# Split the Data into Train ( 80% ) and Test Set ( 20% ).
features_train, features_test, labels_train, labels_test = train_test_split(
    features, one_hot_encoded_labels, test_size=0.2, shuffle=True, random_state=seed_constant)

# Train and Test Data Size
train_size = len(labels_train)
val_size = len(labels_test)

# Data Extraction End
de_time = time.time()
print(f'[INFO] Data Extraction Completed in {((de_time-s_time)/60):.2} Minutes')

if model_type == 'convLSTM':
    print("[INFO] Selected convLSTM Model")
    model = convlstm_model(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST)
    print("[INFO] convLSTM Created Successfully!")
elif model_type == 'LRCN':
    print("[INFO] Selected LRCN Model")
    model = LRCN_model(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST)
    print("[INFO] LRCN Created Successfully!")
else:
    print('[INFO] Model NOT Choosen!!')

# Model Dir
path_to_model_dir = f'{model_type}'
if not os.path.isdir(path_to_model_dir):
    os.makedirs(path_to_model_dir, exist_ok=True)
    print(f'[INFO] Created {path_to_model_dir} Folder')
else:
    print(f'[INFO] {path_to_model_dir} Folder Already Exist')
    f = glob.glob(path_to_model_dir + '/*')
    for i in f:
        os.remove(i)

png_name = f'{model_type}_model_str.png'
path_to_model_str = os.path.join(path_to_model_dir, png_name)
# Plot the structure of the contructed model.
tf.keras.utils.plot_model(model, to_file=path_to_model_str,
           show_shapes=True, show_layer_names=True)
print(f'[INFO] Successfully Created {png_name}')

# Create an Instance of Early Stopping Callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

# Compile the model and specify loss function, optimizer and metrics values to the model
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=["accuracy"])

print(f'[INFO] {model_type} Model Training Started...')

# MLFlow
mlflow.set_experiment('Action Recognition')
with mlflow.start_run(run_name=f'{model_type}_model'):
    mlflow.keras.autolog()

    # Start training the model.
    history = model.fit(x=features_train, y=labels_train, epochs=epochs, batch_size=batch_size,
                        shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback])

    print(f'[INFO] Successfully Completed {model_type} Model Training')

    # Training End
    te_time = time.time()
    print(f'[INFO] Model Training Completed in {((te_time-de_time)/60):.2} Minutes')

    # Evaluate the trained model.
    model_evaluation_history = model.evaluate(features_test, labels_test)

    # Get the loss and accuracy from model_evaluation_history.
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

    # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
    model_file_name = f'{model_type}_model_loss_{model_evaluation_loss:.3}_acc_{model_evaluation_accuracy:.3}.h5'

    # Save your Model
    path_to_save_model = os.path.join(path_to_model_dir, model_file_name)
    model.save(path_to_save_model)
    print(f'[INFO] Model {model_file_name} saved Successfully..')

    # Model Size
    mb_size = os.path.getsize(f'{path_to_save_model}')
    mb_size = round(mb_size / 1e+6, 2)
    print(f'[INFO] {model_type} Model Size: {mb_size} MB')

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
    print(f'[INFO] Successfully Saved {metrics_png_name}')

    # MLFlow Metrics
    mlflow.log_metric('Input Image Size', IMAGE_SIZE)
    mlflow.log_metric('Total Image Data', total_data)
    mlflow.log_metric('Train Size', train_size)
    mlflow.log_metric('Validation Size', val_size)
    mlflow.log_artifact(f'{path_to_metrics}')
    mlflow.log_metric('Model Size MB', mb_size)

    print("[INFO] MLFlow Run: ", mlflow.active_run().info.run_uuid)
mlflow.end_run()

# Total Time
e_time = time.time()
print(f'[INFO] Completed All process in {((e_time-s_time)/60):.2} Minutes')
