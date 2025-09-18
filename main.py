from src.data_loader import DataLoader

loader = DataLoader(data_dir="data/raw/Images")

X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = loader.load_data()

print(f"Classes: {label_encoder.classes_}")
print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
