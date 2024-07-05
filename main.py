import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import save_img
from PIL import Image

st.title("Dog Breed Recognition")
st.subheader("This project uses Deep Learning and MobileNetV2 to detect the breeds of dogs from images. It was "
             "trained using the Stanford Dog Breed dataset.")
st.subheader("The model has 81.12% accuracy on the validation data.")
st.subheader("Upload a photo below to get the model's predictions")
st.caption("Created by Daniyal Dawood")

# File upload
uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])

# Predictions
labels_csv = pd.read_csv("labels.csv")
pathnames = ["train/" + id + ".jpg" for id in labels_csv["id"]]
labels = np.array(labels_csv["breed"])
unique_breeds = np.unique(labels)

loaded_model = tf.keras.models.load_model(
    ("models/20220716-07431657957390-full-image-set-accuracy-mobilenetv2-Adam.h5"),
    custom_objects={'KerasLayer': hub.KerasLayer})

IMG_SIZE = 224


def preprocess_image(path, img_size=IMG_SIZE):
    image = tf.io.read_file(path)

    # Creating RGB values
    image = tf.image.decode_jpeg(image, channels=3)

    # Normalizing RGB values
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize image
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image


# Converting image path and label to tuple
def get_image_label_tuple(path, label):
    image = preprocess_image(path)
    return image, label


BATCH_SIZE = 32


def create_batch(X, y=None, batch_size=BATCH_SIZE, validation=False, test=False):
    if test:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(preprocess_image).batch(BATCH_SIZE)
        return data_batch

    elif validation:
        print("Validation Data")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label_tuple).batch(BATCH_SIZE)
        return data_batch

    else:
        print("Training Data")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data = data.map(get_image_label_tuple)
        data_batch = data.batch(BATCH_SIZE)
        return data_batch


def get_pred_breed(prediciton_probs):
    return unique_breeds[np.argmax(prediciton_probs)]


def plot_custom_pred(prediction_probs, images, n=0):
    pred_prob, image = prediction_probs[n], images[n]

    pred_label = get_pred_breed(pred_prob)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    plt.title((f"Prediction: {pred_label} ({np.max(pred_prob) * 100:2.0f}% confidence)"), color="green")


def plot_custom_pred_confidence(prediction_probs, n=0):
    """
    Plots the 10 most confident predictions
    """

    pred_prob = prediction_probs[n]

    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(top_10_pred_labels)), top_10_pred_values, color="blue")
    plt.xticks(np.arange(len(top_10_pred_labels)), labels=top_10_pred_labels, rotation="vertical")
    return fig


if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        image_array = np.array(image)
        st.write("Uploaded Image")
        st.image(image, use_column_width=True)
    except:
        st.write("The image cannot be displayed, please try again.")

    # Save image to directory
    try:
        save_img("custom/uploaded_image.png", image_array)
        image_path = ["custom/uploaded_image.png"]

        # Get predictions
        custom_data = create_batch(image_path, test=True)
        custom_preds = loaded_model.predict(custom_data)
        custom_pred_label = [get_pred_breed(custom_preds[i]) for i in range(len(custom_preds))]
        st.write(f"The model's most confident prediction is that the dog's breed is **{custom_pred_label[0].capitalize()} "
                 f"({np.max(custom_preds) * 100:2.0f}% confidence)**")
        st.write("Below are the model's 10 most confident predictions")

        col1, col2 = st.columns(2)

        pred_prob = custom_preds[0]
        top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
        top_10_pred_values = pred_prob[top_10_pred_indexes]
        top_10_pred_labels = unique_breeds[top_10_pred_indexes]
        top_10_pred_list = zip(top_10_pred_labels, top_10_pred_values)

        with col1:
            for i in list(top_10_pred_list):
                st.write(f"**{i[0].capitalize()} ({i[1] * 100:2.0f}% confidence)**")

        with col2:
            plotted_preds = plot_custom_pred_confidence(custom_preds)
            st.pyplot(plotted_preds, use_column_width=False)
    except:
        st.write("The app encountered an error. Please try again or upload a different image.")
