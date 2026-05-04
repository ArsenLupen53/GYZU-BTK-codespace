
import numpy as np

"""## Dizi oluşturma fonksiyonları"""

print(f"Aralık ile array tanımla: \n {np.arange(0,10,2)}") #ilk eleman dahil, son eleman dahil değil

print(f"Eşit aralıklar ile array: \n {np.linspace(0,9,6)}") #ilk ve son eleman dahil

print(f"full: \n {np.full((2,4),7)}")

print(f"logspace: \n {np.logspace(0,3,4)}") #10^0 ile 10^3 arasında logaritmik aralık 4 sayı oluştur

print(f"birim matris oluşturucu: \n {np.eye(3)}")

print(f"birim matris olutşru: \n {np.identity(4)}")

print(f"empty: \n {np.empty((2,3))}")

arr = np.array([[1,2,3],[4,5,6]])
empty_like_arr = np.empty_like(arr)
print(f"empty_like: \n {empty_like_arr}")

print(f"tile: \n {np.tile(([1,2,3]),2)}")
print(f"repeat: \n {np.repeat(([1,2,3]),2)}")

"""## İndeksleme ve Dilimleme"""

arr = np.array([10,20,30,40,50])

fancy = arr[[0,2,4]]
print(f"fancy: \n {fancy}")

arr2 = np.array([[10,20],[30,40],[50,60]])
rows = [0,2]
cols = [0]
print(f"seçilen: {arr2[rows,cols]}")

arr = np.array([5,10,15,20,25])
bool_index = arr[arr > 15]
print(f"bool_index: {bool_index}")

"""# View vs Copy"""

arr = np.array([10,20,30,40,50])
arr

slice_view = arr[1:4]
slice_view[:] = 0
print(arr)

arr2 = np.array([10,20,30,40,50])
arr2

slice_copy = arr2[1:4].copy()
slice_copy[:] = 0
print(arr2)
slice_copy

"""#Dizi İşlemleri"""

arr = np.array([1,2,3,4,5])
print(f"min değerin indeksi {np.argmin(arr)}")
print(f"max değerin indeksi {np.argmax(arr)}")

arr = np.array([1,2,3,4,5])
print(f"Kümülatif toplam:  {np.cumsum(arr)}")
print(f"Kümülatif çarpım: {np.cumprod(arr)}")

"""## Şekil İşlemleri"""

arr = np.arange(12)
print(arr)
reshaped = arr.reshape(3,4)
print(reshaped)

from re import A
arr = np.array([[1,2,3],[4,5,6]])
arr
print(f"ravel: {np.ravel(arr)}") #view döner
print(f"flatten: {arr.flatten()}") #copy döner

arr = np.array([[1,2,3],[4,5,6]])
arr
print(f"transpose: {arr.transpose()}")

arr = np.array([1,2,3])
print(arr.shape)
expanded = np.expand_dims(arr,axis=0) # satır ekle
print(expanded.shape)
squeezed = np.squeeze(expanded)
print(squeezed.shape)

#genel birleştirme
a = np.array([[1,2],[3,4]])
b = np.array([[5,6]])

concat = np.concatenate((a,b),axis=0)
print(concat)

#dikey birleştirme
vstacked = np.vstack((a,b))
vstacked

#yatay birleştirme
hstacked = np.hstack((a,np.array([[7,8],[9,10]])))
hstacked

"""## dizileri ayırma"""

arr = np.arange(10)
print(f"split:{np.split(arr,5)}") #5 eşit parçaya ayır

print(f"split:{np.array_split(arr,4)}") #4 parçaya ayır

"""## Rastgele Sayılar"""

#0-1 arasında rastgele sayı üret (uniform dağılım)
print(f"rand: \n {np.random.rand(3,3)}")

# normal dağılıma göre rastgele sayı üret
print(f"randn: \n {np.random.randn(3,3)}")

# belirli bir aralıkta rastgele sayı üret
print(f"randint: \n {np.random.randint(1,10,(3,3))}")

#belirli kümeden rastgele seçmek
arr = np.array([1,2,3,4,5])
print(f"tek seçim: \n {np.random.choice(arr)}")
print(f"çoklu seçim: \n {np.random.choice(arr,size=3,replace=False)}") # replace fonksiyonu aynı karakter seçip seçmediğini belirler

#permütasyon ve shuffle
arr = np.array([1,2,3,4,5])
print(f"permütasyon: \n {np.random.permutation(arr)}") #yeni dizi döndürür
np.random.shuffle(arr) #diziyi yerinde karıştırır
print(f"shuffle: \n {arr}")

#olasılık dağılımları
binom = np.random.binomial(n=10,p=0.5,size=5)
print(binom)
poisson = np.random.poisson(lam=5,size=5)
print(poisson)
normal = np.random.normal(loc=0,scale=1,size=5)
print(normal)
uniform = np.random.uniform(0,1,5)
print(uniform)

#Seed: rastgele sayıların tekrarlanabilir olmasını sağlar.
np.random.seed(42) #bundan sonraki rastgelelikler her çalıştığında farklı değer döndürmez
np.random.randint(0,10,15)

"""## İstatistik ve Veri Analizi"""

data = np.array([12,1,23,31,123,4,1])
from scipy import stats
print(f"mod: {stats.mode(data,keepdims=True).mode[0]}")
print(f"ortalama: {np.mean(data)}")
print(f"medyan: {np.median(data)}")
print(f"varyans: {np.var(data)}")
print(f"standart sapma: {np.std(data)}")

print(f"25.percentil: {np.percentile(data,25)}")
print(f"75.percentil: {np.percentile(data,75)}")

print(f"0.25.percentil: {np.quantile(data,0.25)}")
print(f"0.75.percentil: {np.quantile(data,0.75)}")

#korelasyon ve kovaryans

x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

print(f"korelasyon: \n{np.corrcoef(x,y)}")
print(f"kovaryans: \n{np.cov(x,y)}")

"""## Lineer Cebir"""

a = np.array([1,2,3])
b = np.array([4,5,6])

print(f"dot: {np.dot(a,b)}")
print(f"vdot: {np.vdot(a,b)}")

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print(f"matmul: \n {np.matmul(a,b)}")

a = np.array([[1,2],[3,4]])
det = np.linalg.det(a)
det

a = np.array([[1,2],[3,4]])
inv_a = np.linalg.inv(a)
inv_a

a = np.array([[1,2],[3,4]])
print(f"rank: {np.linalg.matrix_rank(a)}")

a = np.array([[1,2],[3,4]])
eigenvalues, eigenvectors = np.linalg.eig(a)
print(f"eigenvalues: {eigenvalues}")
print(f"eigenvectors: {eigenvectors}")

#lineer denklem çözümü
"""
2x+y=5
x-y=1
"""
a = np.array([[2,1],[1,-1]])
b = np.array([5,1])
solution = np.linalg.solve(a,b)
solution

"""## Maskeler ve Koşullu İşlemler"""

# boolean diziler

arr = np.array([10,20,30,40,50])
mask = arr > 25
print(mask)
print(arr[mask])

#where komutuna göre seçim
arr = np.array([10,20,30,40,50])

result = np.where(arr < 25,0,arr)
result

arr = np.array([10,20,30,40,50])

conditions = [arr < 20, arr < 40, arr >= 40]
choices = ["küçük","orta","büyük"]

labels = np.select(conditions,choices,default="bilinmiyor")
labels

#non zero
arr = np.array([10,20,30,40,50])
print(np.nonzero(arr)) #0 olmayanların indeksini döndürüyor

# any ve all
arr = np.array([10,20,30,40,50])
print(f"Herhangi biri 30dan büyük mü {np.any(arr > 30)}")
print(f"Tüm değerler 30dan büyük mü {np.all(arr > 30)}")

"""## Dosya işlemleri"""

arr = np.array([10,20,30,40,50])
np.save("data.npy",arr)
loaded_arr=np.load("data.npy")
loaded_arr
