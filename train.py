import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. Конфигурация ---
# Пути теперь относительные, предполагая запуск из корня проекта
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, 'server_data', 'RAW DATA') # Путь к папке с изображениями
CSV_DIR = os.path.join(BASE_DIR, 'server_data') # Путь к папке с CSV
MODELS_DIR = os.path.join(BASE_DIR, 'models')

TRAIN_CSV = os.path.join(CSV_DIR, 'train.csv')
VAL_CSV = os.path.join(CSV_DIR, 'validation.csv')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 30
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Класс Датасета ---
class CTScanDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        # Заменяем абсолютные пути на сервере на локальные
        self.filepaths = df['filepath'].apply(lambda x: os.path.join(data_dir, x.split('RAW DATA/')[-1])).values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Ошибка при загрузке изображения {img_path}: {e}")
            # Возвращаем "пустое" изображение и метку, чтобы не прерывать обучение
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.tensor(0.0)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return image, label.unsqueeze(0)

# --- 3. Трансформации и Аугментации ---
# (остаются без изменений)
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# --- 4. Функции обучения и валидации ---
# (остаются без изменений)
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return total_loss / len(dataloader), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# --- 5. Основная функция ---
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Используемое устройство: {DEVICE}")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_dataset = CTScanDataset(train_df, DATA_DIR, transform=get_transforms(is_train=True))
    val_dataset = CTScanDataset(val_df, DATA_DIR, transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = models.efficientnet_b0(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    class_counts = train_df['label'].value_counts()
    if class_counts[1] > 0:
        pos_weight_val = class_counts[0] / class_counts[1]
    else:
        pos_weight_val = 1.0
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_f1 = 0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Эпоха {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: Accuracy={val_metrics['accuracy']:.4f}, Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pth'))
            print(f"Модель сохранена! Новая лучшая F1-мера: {best_val_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Улучшения не было {epochs_no_improve} эпох.")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Ранняя остановка! Нет улучшений в течение {EARLY_STOPPING_PATIENCE} эпох.")
            break

if __name__ == '__main__':
    main()
