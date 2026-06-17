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

