import cv2
import numpy as np
import os


def frames_extraction(video_path, SEQUENCE_LENGTH, IMAGE_SIZE, yolov7_model, yolov7_conf):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []

    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES,
                         frame_counter * skip_frames_window)

        # Reading the frame from the video.
        success, frame = video_reader.read()

        # Check if Video frame is not successfully read then break the loop
        if not success:
            print('[INFO] Video Reading Failed or Video Ended..')
            break

        bbox_list = []
        # Action - ROI
        results = yolov7_model(frame)
        # Bounding Box
        box = results.pandas().xyxy[0]

        for i in box.index:
            xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(
                box['xmax'][i]), int(box['ymax'][i]), box['confidence'][i]
            bbox_list.append([xmin, ymin, xmax, ymax, conf])

        if len(bbox_list) > 0:
            # for bbox in bbox_list:
            # Only taking One bbox
            for bbox in bbox_list:
                if bbox[4] > yolov7_conf:
                    bbox1 = bbox_list[0]

                    frame_roi = frame[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
                    # Resize the Frame to fixed height and width.
                    resized_frame = cv2.resize(
                        frame_roi, (IMAGE_SIZE, IMAGE_SIZE))

                    # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
                    normalized_frame = resized_frame / 255

                    # Append the normalized frame into the frames list
                    frames_list.append(normalized_frame)
                    break

    # Release the VideoCapture object.
    video_reader.release()

    # Return the frames list.
    return frames_list


def create_dataset(CLASSES_LIST, DATASET_DIR, SEQUENCE_LENGTH, IMAGE_SIZE, yolov7_model, yolov7_conf):
    '''
    This function will extract the data of the selected classes and create the required dataset.
    Returns:
        features:          A list containing the extracted frames of the videos.
        labels:            A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    '''

    # Declared Empty Lists to store the features, labels and video file path values.
    features = []
    labels = []
    video_files_paths = []

    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):

        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')

        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        # Iterate through all the files present in the files list.
        for file_name in files_list:

            # Get the complete video path.
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

            # Extract the frames of the video file.
            frames = frames_extraction(
                video_file_path, SEQUENCE_LENGTH, IMAGE_SIZE, yolov7_model, yolov7_conf)

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.

            if len(frames) < SEQUENCE_LENGTH:
                print(
                    f'[INFO] Length of Frame Sequence in your Data is: {len(frames)} Less than given Sequence Lenght: {SEQUENCE_LENGTH}')
                print(
                    '[INFO] Change the Data or Reduce Sequence Lenght (NOT Recommended)')
                break

            if len(frames) == SEQUENCE_LENGTH:

                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths
