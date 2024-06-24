import tensorflow as tf
import json
import numpy as np

def print_model_info(model, compile_info = False):
    model.summary(show_trainable = True)
    config = model.get_compile_config()
    print(json.dumps(config, indent = 4))
    print()


def compile(model, optimizer, loss, metrics):
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

def evaluate(model, x_test, y_test):
    return model.evaluate(x_test, y_test, verbose = 2)

def evaluate_print(model, x_test, y_test):
    print(model.evaluate(x_test, y_test, verbose = 2, return_dict = True))

def predict(model, x, softmax = False):
    preds = model.predict(x)
    if softmax == False:
        return preds
    else:
        return tf.nn.softmax(preds)

def preds_to_class(preds):
    return np.argmax(preds, axis = 1)
