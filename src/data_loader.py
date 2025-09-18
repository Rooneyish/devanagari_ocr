import os 
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_dir, img_size= (32,32), train_size = 0.8, val_size = 0.1, test_size = 0.1, random_state = 42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def load_data(self):
        images = []
        labels = []

        for class_name in sorted(os.listdir(self.data_dir)):
            class_path = os.path.join(self.data_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue

            img_files = os.listdir(class_path)
            for img_name in img_files:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.img_size)
                images.append(img)
                labels.append(class_name)

        X = np.array(images, dtype='float32') / 255.0
        X = np.expand_dims(X, -1)
        y = self.label_encoder.fit_transform(labels)

        print(f"Dataset loaded: {X.shape[0]} samples, {len(self.label_encoder.classes_)} classes.")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size+self.val_size, random_state=self.random_state, stratify=y
        )

        relative_val_size = self.val_size/ (self.test_size+self.val_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size= 1-relative_val_size, random_state=self.random_state, stratify=y_temp
        )

        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, self.label_encoder