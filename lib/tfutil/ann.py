import tensorflow as tf

def build_ann(input_shape, output_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = input_shape),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_size)
    ])

    return model

