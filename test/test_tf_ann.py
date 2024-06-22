from lib.tfutil.ann import build_ann
from lib.dataset.tf_dataset import load_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_mnist()
    model = build_ann(x_train.shape[1:], to_categorical(y_test).shape[1])

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(model)
