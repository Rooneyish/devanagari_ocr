import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import DataLoader
from model_cnn import custom_cnn_model

def train():
    loader = DataLoader(data_dir="/home/rooneyish/Documents/projects/devanagiri_ocr/data/raw/Images", img_size=(32,32), val_size= 0.1, test_size=0.1)

    X_train, X_val, X_test, y_train, y_val, y_test, classes = loader.load_processed_data()

    model = custom_cnn_model(input_shape=X_train.shape[1:], num_classes=len(classes.classes_))

    history=model.fit(X_train, y_train,epochs = 100, batch_size = 64, validation_data = (X_val,y_val))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
    print(f'Test Accuracy: {test_acc: .4f}, Test Loss = {test_loss: .4f}')

    save_dir = "/home/rooneyish/Documents/projects/devanagiri_ocr/experiments/cnn_baseline"
    os.makedirs(save_dir, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(save_dir, "training_logs.csv"), index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

if __name__ == "__main__":
    train()