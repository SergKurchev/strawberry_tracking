# Strawberry Tracker & Pose Estimator

Система для отслеживания клубники и восстановления позы камеры на основе глубокого обучения (YOLOv8) и глобальных признаков (DISK/LightGlue).

## Основные возможности

- **Детекция и сегментация**: Использование YOLOv8 для поиска ягод.
- **Глобальная локализация**: Восстановление поворота и смещения камеры по фоновым ориентирам (landmarks).
- **Устойчивость к паузам**: Встроенный детектор малых перемещений (Small Motion Fallback) для предотвращения ошибок при остановке робота.
- **Галерея дескрипторов**: Сохранение идентификаторов (ID) ягод даже при частичном перекрытии листвой.

## Структура проекта

- `strawberry_tracker.py`: Основной модуль с классом `StrawberryTracker`.
- `config.yaml`: Файл конфигурации (пороги, пути, параметры камеры).
- `test_on_coco.py`: Скрипт для тестирования на наборах данных (COCO-format).
- `run_tracker.py`: Пример интеграции в реальном времени.

## Быстрый старт (Установка)

Проект использует менеджер [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run python test_on_coco.py
```

## Использование StrawberryTracker

### Инициализация

```python
from strawberry_tracker import StrawberryTracker
from ultralytics import YOLO
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

yolo = YOLO(config['yolo_weights_path'])
tracker = StrawberryTracker(yolo, device='cuda', config=config)
```

### Основной метод Forward

Метод `forward` обрабатывает входящий кадр и возвращает все данные для навигации и трекинга.

```python
K - матрица внутренних параметров камеры (3x3)
R, t, ids, boxes, matches1, matches2, inliers1, inliers2, debug = tracker.forward(img_rgb, K)
```

**Выходные данные:**

- `R`, `t`: Матрица поворота и вектор смещения (относительно предыдущего кадра).
- `ids`: Список глобальных ID для каждой найденной ягоды.
- `boxes`: Координаты bounding boxes `[x1, y1, x2, y2]`.
- `inliers1`, `inliers2`: Проверенные ключевые точки (для визуализации позы).
- `debug`: Статистика (количество инлайеров, норма смещения).

## Настройка параметров (config.yaml)

- `yolo_conf`: Порог уверенности для ягод (рекомендуется `0.5` для исключения ложных срабатываний).
- `min_motion_thresh`: Порог (в пикселях), ниже которого камера считается статичной (предотвращает ошибки RANSAC).
- `ransac_threshold`: Допустимая ошибка репроекции в пикселях.
- `pose_solver`: Выбор алгоритма (`RANSAC` или `MAGSAC`).

## Нюансы и рекомендации

1. **Матрица K**: Крайне важно прописать точные параметры фокусного расстояния (`focal_length`) и центра (`center_x/y`) в конфиге. Без этого восстановление `R` и `t` будет неточным.
2. **Освещение**: Глобальные признаки (DISK) устойчивы к изменениям света, но при сильных бликах на листьях количество инлайеров может снижаться.
3. **Статичные кадры**: Если `inliers` на графике подкрашены красным, а в заголовке написано `STATIC`, значит движение слишком мало для вычисления честной позы, и алгоритм сохранил предыдущую ориентацию.


## Экспорт данных (Export Data)

В `config.yaml` доступен режим `export_data: true`. При его активации `test_on_coco.py` автоматически сохраняет всю информацию о сцене в папку `exported_data/`:
1. `poses.csv`: Матрицы поворота ($R$) и векторы смещения ($t$) для вычисления одометрии.
2. `tracking.json`: Идентификаторы (ID) ягод и их bounding boxes на каждом кадре (идеально для метрик MOTA/IDF1).
3. `inliers.json`: Точные 2D-координаты совпавших ключевых точек, по которым была рассчитана матрица $R$.

### Пример загрузки и анализа данных (Python)

```python
import json
import pandas as pd
import numpy as np

# 1. Загрузка траектории камеры (Одометрия)
poses_df = pd.read_csv('exported_data/poses.csv')
print("Первые 5 кадров одометрии:")
print(poses_df.head())

# Конвертация строки CSV обратно в матрицу вращения R (3x3)
row = poses_df.iloc[0]
R = np.array([
    [row['r11'], row['r12'], row['r13']],
    [row['r21'], row['r22'], row['r23']],
    [row['r31'], row['r32'], row['r33']]
])

# 2. Загрузка трекинга ягод
with open('exported_data/tracking.json', 'r') as f:
    tracking = json.load(f)

# Простая операция: Найти на скольких кадрах появлялась ягода с ID 0
berry_0_frames = []
for frame_name, detections in tracking.items():
    for det in detections:
        if det['id'] == 0:
            berry_0_frames.append(frame_name)
            
print(f"\\nЯгода ID 0 была успешно отслежена на {len(berry_0_frames)} кадрах.")