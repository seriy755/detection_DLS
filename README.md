# Проект для Deep Learning School
## Детекция на датасете игральных карт с помощью детектора SSD
---
### Описание:

Данный репозиторий является финальным проектом от [Deep Learning School](https://www.dlschool.org/)

Был выбран проект по детекции объектов на изображении.

**2 сценарий:**
* Был выбран детектор SSD300 из [Torcvision.ssd300_vgg16](https://pytorch.org/vision/master/generated/torchvision.models.detection.ssd300_vgg16.html)
* Взят датасет игральных карт: [оригинальный репо](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

Обученная модель может быть применена, например для выявления шулерства в казино. Можно встроить данную модель в детектор слежения, который отслеживает карты на руках участников, и при шулерстве(например, одна из карт была заменена) сообщает об этом.

### Запуск:

#### 1. Демо:
* Запускаем файл [`demo.ipynb`](https://github.com/seriy755/detection_DLS/blob/main/demo.ipynb)
* Передаем в функцию `demo` модель, тензор изображения и порог

### 2. Тренировка:
* Запускаем файл [`train.ipynb`](https://github.com/seriy755/detection_DLS/blob/main/train.ipynb)
* Объявляем модель, оптимизатор, его параметры, планировщик и т.д.
* Передаём в функцию `train` модель, оптимизатор, его параметры, количество эпох, train/test-loaderы, device, планировщик, его параметры
 
Во время обучения графики потерь выводятся на экран при помощи библиотеки [tensorboard](https://github.com/tensorflow/tensorboard)

### 3. Оценка модели:
* Запускаем файл [`eval.ipynb`](https://github.com/seriy755/detection_DLS/blob/main/eval.ipynb)
* Передаём в функцию `eval` модель, датасет, порог(для Precision, Recall, F1)

### Полученные результаты:
В результате обучения модели мне удалось получить следующие метрики(усредненные по все классам):
| Precision | Recall | F1 | AP50 | mAP |
|-------|:-----:|:-----:|:-----:|:-----:|
| 39.46% | 34.18% | 35.94% | 35.62% | 18.43% |

## Пример:
### Оригинал
![Origin](https://github.com/seriy755/detection_DLS/blob/main/example.jpg)

### С предсказнными boxами:
![Predict](https://github.com/seriy755/detection_DLS/blob/main/prediction.png)
