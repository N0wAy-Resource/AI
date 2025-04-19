import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示WARNING和ERROR
from keras._tf_keras.keras import layers, models
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 数据加载和预处理
def load_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for label_id, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label_id)

    return np.array(images), np.array(labels), class_names


# 数据集路径
DATA_DIR = r"trainface\\"
images, labels, class_names = load_data(DATA_DIR)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# 数据归一化
X_train = X_train / 255.0
X_test = X_test / 255.0

# 创建数据增强生成器
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# 构建CNN模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),#
    layers.Dropout(0.5),
    layers.Dense(41, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=(X_test, y_test),
)

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.2%}")

# 保存模型
model.save('face_recognition_cnn.keras')

# 创建类别参考图像字典
class_reference_images = {}
for class_id in range(len(class_names)):
    class_indices = np.where(y_train == class_id)[0]
    if len(class_indices) > 0:
        class_reference_images[class_id] = X_train[class_indices[0]]

# 保存元数据（与模型同名，扩展名为.npz）
np.savez('face_recognition_cnn_metadata.npz',
         class_names=class_names,
         class_reference_images=class_reference_images)

print(f"\n训练完成，数据已保存。")
