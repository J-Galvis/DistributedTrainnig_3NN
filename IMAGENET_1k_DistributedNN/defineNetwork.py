from datasets import load_dataset
from keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)

dataset_train = load_dataset(
    "imagenet-1k",
    split="train",
    streaming=False,
    token="<my_token>",
    trust_remote_code=True,
)
