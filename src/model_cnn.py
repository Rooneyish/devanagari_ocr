import tensorflow as tf
from tf.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tf.keras import models

def custom_cnn_model(input_shape, num_classes):
    model = models.Sequential(
        [
            Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape=(32,32,1)),
            MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'valid'),
            Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu'),
            MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'valid'),
            Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu'),
            MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'valid'),
            Flatten(),
            Dense(1024, activation = 'relu'),
            Dense(512, activation = 'relu'),
            Dense(256, activation = 'relu'),
            Dense(num_classes, activation = 'softmax'),
        ]
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model