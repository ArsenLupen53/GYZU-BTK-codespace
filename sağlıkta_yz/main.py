"""
Proje: Deri kanser sınıflandırma projesi (iyi huylu ve kötü huylu)
Yöntem: Logistic Regression ve KNN + HOG Feature extraction + sklearn

Amaç: 
    - Bu projede amacımız, cilt lezyonu görüntülerinden HOG (Histogram of Oriented Gradients) öznitelik çıkararak logistic regression ile sınıflandırma yapmak
    - hedef, lezyonun "benign" -> iyi huylu mu yoksa "malignant" -> kötü huylu mu olduğunu tahmin etmek (binary classification problemi çözme)

Veri seti:
    kaggle: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign
    klasör yapısı:
        - train
            - benign
            - malignant

Kullanılacak olan yöntem:
    - Görüntüleri gri tona çevirme
    - 128x128 boyutuna resize etme
    - HOG ile feature extraction (öznitelik çıkarma)
    - Logistic regression modeli ile sınıflandırma

Akış:
    1. görüntüleri yükle
    2. hog özniteliklerini çıkart
    3. train-test split
    4. normalize etme
    5. logistic regression eğit
    6. performansı ölç
    7. confusion matrix ile değerlendirme
    8. örnek tahminleri görselleştir

gerekli kütüphaneleri kur:
pip install scikit-learn scikit-image numpy matplotlib opencv-python
"""

import os
import cv2 # open cv görüntü işleme kütüphanesi
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #ölçekleme
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from skimage.feature import hog # feature extraction

#veri yükleme fonksiyonu
def load_images_from_folder(folder_path,label):
    X = [] #bağımsız değişkenler
    y = [] #target variable

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path) # resmin dosya yoluna göre resmi oku

        if img is None:
            continue
        
        # bgr2gray: renkli görüntüden siyah beyaza yani gri tonlamaya dönüşüm yap

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resimlerin boyutunu yeniden ayarla 224x224 -> 128x128
        gray = cv2.resize(gray, (128,128))

        # HOG: Histogram of oriented gradients feature extraction yöntemi
        features, hog_image = hog(
            gray, # img
            orientations=9, #kenar yönü
            pixels_per_cell=(8,8), # her hücrenin 8x8 olduğunu gösterir
            cells_per_block=(2,2), # normalizasyon
            block_norm = "L2-Hys", # normalizasyon yöntemi
            visualize = True # HOG görüntüsünü görselleştir
        )

        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

#klasör yolları
base_path = "sağlıkta_yz/train"
benign_folder = os.path.join(base_path, "benign")
malignant_folder = os.path.join(base_path, "malignant")
print("Veriler yükleniyor")

# verileri yükle
X_benign, y_benign = load_images_from_folder(benign_folder, 0) # iyi huylu -> 0
X_malignant, y_malignant = load_images_from_folder(malignant_folder, 1) # kötü huylu -> 1

# verileri birleştir
X = np.vstack([X_benign, X_malignant])
y = np.hstack([y_benign, y_malignant]) 

print("Veriler yüklendi")


# veri setinin görselleştirilmesi
sample_images = [] #görüntüler için liste
sample_labels = [] #etiketler için liste

# benign klasöründen 3, malignant klasöründen ise 3 adet görüntü al
benign_files = os.listdir(benign_folder)[:3]
malignant_files = os.listdir(malignant_folder)[:3]

for img_name in benign_files:
    img = cv2.imread(os.path.join(benign_folder, img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    sample_images.append(gray)
    sample_labels.append("Benign")

for img_name in malignant_files:
    img = cv2.imread(os.path.join(malignant_folder, img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    sample_images.append(gray)
    sample_labels.append("Malignant")

# plotlama
plt.figure(figsize=(10,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(sample_images[i], cmap="gray")
    plt.title(sample_labels[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# veri setini train ve test olmak üzere 2ye ayrılması
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # stratify: sınıf oranlarının bozulmamasını sağlar

# normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# model training
# logistic regression modeli eğitimi
log_reg_model = LogisticRegression(solver="liblinear")
log_reg_model.fit(X_train_scaled, y_train)

# knn modeli
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

