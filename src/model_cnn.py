import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from data_loader import DataLoader

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
            Dense(128, activation = 'relu'),
            Dense(num_classes, activation = 'softmax'),
        ]
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(project_root)
    summary_path = os.path.join(project_root, "experiments/model_summary.txt")
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    print('Model saved at experiments folder as model_summary.txt')

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model

if __name__ == "__main__":
    loader = DataLoader(data_dir="/home/rooneyish/Documents/projects/devanagiri_ocr/data/raw/Images", img_size=(32,32), val_size= 0.1, test_size=0.1)
    X_train, X_val, X_test, y_train, y_val, y_test,label_encoder = loader.load_data()
    custom_cnn_model(input_shape=X_train.shape[1:], num_classes=len(label_encoder.classes_))
