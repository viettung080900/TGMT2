import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# 1. Tải và tiền xử lý ảnh
def load_and_preprocess_image(image_path, size=(64, 64)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Cannot load image at {image_path}")
        return None
    image = cv2.resize(image, size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.flatten()


# 2. Tạo tập dữ liệu mẫu (giả định)
image_paths = [

    'images/meo.png',
    'images/meo2.png',
    'images/hoa1.png',
    'images/hoa2.png'
]
labels = [
    'flower',
    'flower',
    'animal',
    'animal'
]

data = []
valid_labels = []
for img, label in zip(image_paths, labels):
    processed_image = load_and_preprocess_image(img)
    if processed_image is not None:
        data.append(processed_image)
        valid_labels.append(label)

if len(data) == 0:
    print("Error: No images were loaded successfully. Please check the image paths.")
else:
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(valid_labels)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.3, random_state=42)

    classifiers = {
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(n_neighbors=1),
        'Decision Tree': DecisionTreeClassifier()
    }

    results = {}
    predictions = {}
    for name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = clf.predict(X_test)
        predictions[name] = y_pred
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

        results[name] = {
            'Time (s)': training_time,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        }
        print(f"Model: {name}\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}\n")

    print("So sánh hiệu năng giữa các thuật toán:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # 6. Hiển thị hình ảnh và kết quả phân loại
    num_images = len(image_paths)
    num_models = len(classifiers)

    plt.figure(figsize=(10, 5 + 2 * num_models))

    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        plt.subplot(num_models + 1, num_images, i + 1)
        plt.imshow(original_image)
        plt.title(f"Original: {label}")
        plt.axis('off')

        for j, name in enumerate(classifiers.keys()):
            if i < len(predictions[name]):
                predicted_label = label_encoder.inverse_transform([predictions[name][i]])[0]
                plt.subplot(num_models + 1, num_images, num_images + i + 1 + j * num_images)
                plt.imshow(original_image)
                plt.title(f"{name}:\n{predicted_label}")
                plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 7. Lưu thông tin vào file README.md
    with open('README.md', 'w') as f:
        f.write("# Kết quả phân loại hình ảnh\n\n")
        f.write("## Thông tin mô hình\n")
        for name, metrics in results.items():
            f.write(f"### Mô hình: {name}\n")
            for metric, value in metrics.items():
                f.write(f"  - {metric}: {value:.4f}\n")
            f.write("\n")

        f.write("## Nhãn tương ứng với hình ảnh\n")
        for img_path, label in zip(image_paths, labels):
            f.write(f"- Hình ảnh: {img_path}, Nhãn: {label}\n")