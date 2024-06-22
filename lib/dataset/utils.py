from tensorflow.keras.utils import to_categorical

def onehot_encoding(y, num_classes = None):
    return to_categorical(y, num_classes)

def get_classes(y, is_onehot_encoding = True):
    if(is_onehot_encoding):
        return y.shape[1]
    else:
        return onehot_encoding(y).shape[1]
