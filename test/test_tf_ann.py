import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from lib.tfutil.ann import build_ann
from lib.tfutil.utils import compile, print_model_info, evaluate, evaluate_print, predict, preds_to_class
from lib.dataset.tf_dataset import load_fashion_mnist, load_mnist
from lib.dataset.utils import onehot_encoding, get_classes

import numpy as np
import tensorflow as tf

def test_load_mnist():
    x_train, y_train, x_test, y_test = load_mnist()
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000, )
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000, )
    assert np.max(x_train) == 1.0
    assert np.min(x_train) == 0
    assert np.max(x_test) == 1.0
    assert np.min(x_test) == 0
    assert get_classes(y_test, False) == 10

def test_load_fasion_mnist():
    x_train, y_train, x_test, y_test = load_fashion_mnist()
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000, )
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000, )
    assert np.max(x_train) == 1.0
    assert np.min(x_train) == 0
    assert np.max(x_test) == 1.0
    assert np.min(x_test) == 0
    assert get_classes(y_test, False) == 10

def test_ann():
    x_train, y_train, x_test, y_test = load_mnist()
    model = build_ann(x_train.shape[1:], get_classes(y_test, False))
    assert model(x_train[0:1]).shape == (1, 10)

    compile(model, 'adam', tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), ['accuracy'])
    print_model_info(model, True)

    #model.fit(x_train, y_train, epochs = 2)

    evaluate_print(model, x_test, y_test)
    preds = predict(model, x_test)
    preds_softmax = predict(model, x_test, True)
    preds_class = preds_to_class(preds_softmax)
    assert preds.shape == (10000, 10)
    assert preds_class.shape == (10000, )
    assert preds_softmax.shape == (10000, 10)
