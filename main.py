import tensorflow as tf
from tf_models import DCGAN
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 56


def main():
    # https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models
    cat_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "./datasets/catface_dataset/",
        labels="inferred",
        label_mode=None,
        color_mode="grayscale",
        batch_size=32,
        image_size=(IMG_SIZE, IMG_SIZE),  # img_size should be 28 for now
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )

    print(len(cat_dataset))

    dcgan = DCGAN.Dcgan()
    dcgan.train(cat_dataset, 200)


if __name__ == "__main__":
    main()
