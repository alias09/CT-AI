import sys
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

# Константы (должны совпадать с теми, что были в Ruby)
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(image_path):
    """Загружает, изменяет размер и нормализует изображение."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    # Конвертируем в numpy array и нормализуем до [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Стандартизация
    img_array = (img_array - MEAN) / STD
    
    # HWC -> CHW (Height, Width, Channel -> Channel, Height, Width)
    img_array = img_array.transpose(2, 0, 1)
    
    # Добавляем batch dimension: CHW -> NCHW (1, C, H, W)
    return np.expand_dims(img_array, axis=0)

def sigmoid(x):
    """Вычисляет сигмоиду для преобразования логита в вероятность."""
    return 1 / (1 + np.exp(-x))

def run_inference(image_dir, model_path):
    """Запускает модель на всех PNG в директории и возвращает результат."""
    try:
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name

        image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        
        if not image_files:
            return {"status": "error", "message": "No PNG files found in the directory."}

        max_pathology_prob = 0.0
        slices_data = []
        
        for image_path in image_files:
            input_tensor = preprocess_image(image_path)
            
            # Запускаем инференс
            result = sess.run(None, {input_name: input_tensor})
            
            # Модель возвращает логит. Преобразуем его в вероятность с помощью сигмоиды.
            logit = result[0][0][0]
            pathology_prob = sigmoid(logit)

            # Собираем данные по каждому срезу
            slices_data.append({
                "filename": os.path.basename(image_path),
                "probability": float(pathology_prob)
            })
            
            if pathology_prob > max_pathology_prob:
                max_pathology_prob = pathology_prob

        return {
            "status": "success",
            "pathology_detected": bool(max_pathology_prob > 0.5),
            "max_probability": float(max_pathology_prob), 
            "slices_data": slices_data 
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(json.dumps({"status": "error", "message": "Usage: python run_inference.py <image_directory> <model_path>"}))
        sys.exit(1)

    image_directory = sys.argv[1]
    model_path = sys.argv[2]

    results = run_inference(image_directory, model_path)
    print(json.dumps(results))
