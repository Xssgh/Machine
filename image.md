# 텐서플로우 기초 사용법 정리

## 1. 텐서플로우 임포트

```python
import tensorflow as tf
# 텐서플로우를 tf라는 이름으로 불러옴

from tensorflow.keras import datasets, layers, models
# 데이터셋 제공, 모델의 층 구성, 모델을 생성하고 구성함

import matplotlib.pyplot as plt
# 데이터 시각화 라이브러리
```

> 가장 기본적으로 사용되는 코드로서 대부분의 텐서플로우 초입에 사용됨

---

## 2. 합성곱 층 만들기

```python
model = models.Sequential()
# 시퀀셜 모델 생성 (층을 쌓을 수 있는 신경망 구조)

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 필터, 커널 개수 -> 32개의 특징 맵 생성
# 각 필터의 크기는 3x3
# 입력 이미지의 형태는 32x32 크기의 RGB컬러 이미지

model.add(layers.MaxPooling2D((2, 2)))
# 2x2 영역에서 가장 큰 값을 선택해서 출력
# 입력 이미지 크기를 반으로 줄임

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 필터는 64개, 각 필터는 3x3
# 이전 층에서 추출된 정보로 더 복잡한 패턴 학습

model.add(layers.MaxPooling2D((2, 2)))
# 다시 입력 이미지 압축

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 64개의 필터에 3x3의 필터 크기
```

---

## 3. Dense 층 추가하기

```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
```

---

# 이미지 분류 관련 코드

## 1. 데이터 시각화 (AI가 이미지를 직접 봄)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
```

---

## 2. 이미지의 크기 등 데이터 확인

```python
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
```

---

## 3. 훈련 결과 시각화하기

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

---

## 4. 데이터 증강 1

> 이미지를 수평으로 뒤집고, 회전시키고, 확대/축소하는 등의 데이터 증강 기법을 적용한 Keras 모델 정의

```python
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
```

---

## 5. 데이터 증강 2 (시각화 포함)

> 훈련 데이터에서 하나의 배치를 가져와서 9개의 이미지를 데이터 증강 기법으로 변형 후 시각화

```python
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
```

## 6. 최상위층 고정 해제

```python
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
```

