import os 
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_dir, img_size= (32,32), val_size = 0.1, test_size = 0.1, random_state = 42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def load_data(self):
        images = []
        labels = []

        classes_found = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        print(f"Classes Found: {classes_found}")

        for class_name in classes_found:
            class_path = os.path.join(self.data_dir, class_name)
            img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png'))]

            if not img_files:
                print(f'Imange file not found at {class_path}')
                continue

            for img_name in img_files:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f'Cant read {img_path}')
                    continue
                img = cv2.resize(img, self.img_size)
                images.append(img)
                labels.append(class_name)

            print(f'{class_name}: {len(img_files)} files is loaded')

        if len(images) == 0:
            raise ValueError(f"No images loaded from {self.data_dir}. Check paths and image files!")

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

        processed_dir = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(processed_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(processed_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
        np.save(os.path.join(processed_dir, 'classes.npy'), self.label_encoder.classes_)    

        print(f'Processed Images saved at {processed_dir}')

        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, self.label_encoder

    def load_processed_data(self):
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed')

        file_names = ['X_train.npy', 'X_val.npy', 'X_test.npy', 'y_train.npy', 'y_val.npy', 'y_test.npy', 'classes.npy']

        if all(os.path.exists(os.path.join(processed_dir, file)) for file in file_names):
            X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
            X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
            X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
            y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
            y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))
            y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
            classes = np.load(os.path.join(processed_dir, 'classes.npy'))
        else:
            loader = DataLoader(data_dir="/home/rooneyish/Documents/projects/devanagiri_ocr/data/raw/Images", img_size=(32,32), val_size= 0.1, test_size=0.1)
            X_train, X_val, X_test, y_train, y_val, y_test, classes = loader.load_data()
        
        return X_train, X_val, X_test, y_train, y_val, y_test, classes
