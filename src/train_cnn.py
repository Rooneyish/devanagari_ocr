import os
import tensorflow as tf
from data_loader import DataLoader
from model_cnn import custom_cnn_model

def train():
    
    loader = DataLoader(data_dir="/home/rooneyish/Documents/projects/devanagiri_ocr/data/raw/Images", img_size=(32,32), val_size= 0.1, test_size=0.1)
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = loader.load_data()

    model = custom_cnn_model(input_shape=X_train.shape[1:], num_classes=len(label_encoder.classes_))

    history=model.fit(X_train, y_train,epochs = 25, batch_size = 64, validation_data = (X_val,y_val))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
    print(f'Test Accuracy: {test_acc: .4f}, Test Loss = {test_loss: .4f}')

    import pandas as pd
    save_dir = "/home/rooneyish/Documents/projects/devanagiri_ocr/experiments/cnn_baseline"
    os.makedirs(save_dir, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(save_dir, "training_logs.csv"), index=False)


if __name__ == "__main__":
    train()