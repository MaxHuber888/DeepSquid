import tensorflow as tf
import numpy as np
import cv2
import os
import gradio as gr
from keras.models import load_model
from pytube import YouTube
import pickle


def load_video_from_url(youtube_url):
    visible = True
    try:
        # DOWNLOAD THE VIDEO USING THE GIVEN URL
        yt = YouTube(youtube_url)
        yt_stream = yt.streams.filter(file_extension='mp4').first()
        title = yt_stream.title
        src = yt_stream.download()
        capture = cv2.VideoCapture(src)

        # SAMPLE FRAMES FROM VIDEO FILE
        sampled_frames = sample_frames_from_video_file(capture)

        # PICK EXAMPLE FRAME FROM THE MIDDLE OF THE SAMPLED FRAMES
        example_frames = [
            sampled_frames[len(sampled_frames) // 4],
            sampled_frames[len(sampled_frames) // 2],
            sampled_frames[3 * len(sampled_frames) // 4],
        ]

        # SWAP COLOR CHANNELS FOR EXAMPLE FRAMES
        example_frames = swap_color_channels(example_frames)

        # DELETE VIDEO FILE
        if os.path.exists(src):
            os.remove(src)

        # CONVERT SAMPLED FRAMES TO TENSOR
        frames_tensor = tf.expand_dims(tf.convert_to_tensor(sampled_frames, dtype=tf.float32), axis=0)

        # SAVE TENSOR TO FILE
        pickle.dump(frames_tensor, open("frames_tf.pkl", "wb"))

    except Exception as e:
        title = "Error while loading video: " + str(e)
        visible = False
        example_frames = [np.zeros((256, 256, 3)) for _ in range(3)]

    # Define visible prediction components to show upon video loaded
    predVideoBtn = gr.Button(value="Classify Video", visible=visible)

    predOutput = gr.Label(
        label="DETECTED LABEL (AND CONFIDENCE LEVEL)",
        num_top_classes=2,
        visible=visible
    )

    return title, example_frames, predVideoBtn, predOutput


def detect_deepfake():
    # LOAD FRAMES
    frames_tf = pickle.load(open("frames_tf.pkl", "rb"))

    # DELETE FRAMES FILE
    if os.path.exists("frames_tf.pkl"):
        os.remove("frames_tf.pkl")

    # LOAD THE RNN MODEL FROM DISK
    loaded_model = load_model("MesonetRNN.keras")
    # loaded_model.summary()

    # GET PREDICTION
    out = loaded_model.predict(frames_tf)
    real_confidence = out[0][0]
    fake_confidence = 1 - real_confidence
    confidence_dict = {"FAKE": fake_confidence, "REAL": real_confidence}

    # MAKE FLAG BUTTON VISIBLE
    flagBtn = gr.Button(value="Flag Output", visible=True)

    # RETURN THE OUTPUT LABEL AND EXAMPLE FRAMES
    return confidence_dict, flagBtn


def sample_frames_from_video_file(capture, sample_count=10, frames_per_sample=10, frame_step=10,
                                  output_size=(256, 256)):
    # Read each video frame by frame
    result = []

    video_length = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (frames_per_sample - 1) * frame_step

    max_start = video_length - need_length

    sample_starts = []

    for sample in range(sample_count):
        sample_start = int(max_start * sample / sample_count)
        sample_starts.append(sample_start)
        # print(sample_start)

    for start in sample_starts:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        # ret is a boolean indicating whether read was successful, frame is the image itself
        ret, frame = capture.read()
        result.append(format_frames(frame, output_size))

        for _ in range(frames_per_sample - 1):
            for _ in range(frame_step):
                ret, frame = capture.read()
            if ret:
                frame = format_frames(frame, output_size)
                result.append(frame)
            else:
                result.append(np.zeros_like(result[0]))
    capture.release()

    return np.array(result)


def swap_color_channels(frames):
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]


def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame
