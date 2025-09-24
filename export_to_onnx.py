import torch
import torchvision.models as models
import os

# --- 1. Конфигурация для локальной машины ---
BASE_DIR = '/Users/kvp/Documents/ИИ проекты/ИИ-КТ 2.0'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, 'model.onnx')

IMG_SIZE = 224
DEVICE = 'cpu' # Для экспорта всегда лучше использовать CPU

# --- 2. Основная функция экспорта ---
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Файл модели не найден по пути: {MODEL_PATH}")
        return

    print(f"Загрузка модели из: {MODEL_PATH}")

    # Загрузка архитектуры модели
    model = models.efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    
    # Загрузка обученных весов
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Переводим модель в режим оценки - это ВАЖНО для экспорта

    # Создаем "пустышку" - пример входных данных для модели
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    print(f"\nНачинаю экспорт в ONNX формат...")
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'}}
    )

    print(f"\nМодель успешно экспортирована и сохранена в: {ONNX_MODEL_PATH}")
    if os.path.exists(ONNX_MODEL_PATH):
        print("Файл model.onnx успешно создан!")
    else:
        print("Ошибка: Файл model.onnx не был создан.")

if __name__ == '__main__':
    main()
