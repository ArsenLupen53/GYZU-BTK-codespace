
"""
## Pandasa giriş
"""

import pandas as pd

#seriler tek boyutludur
s = pd.Series([10,20,30,40],index = ["a","b","c","d"])
s

# dataframeler iki boyutludur
data = {
    "isim": ["ali","ayşe","mehmet"],
    "yas": [25,30,22],
    "sehir": ["istanbul","ankara","izmir"]
}
df = pd.DataFrame(data)
df

print(f"Şekil (satır,sütün): {df.shape}")
print(f"Sütün isimleri: {df.columns}")
print(f"İndeksler: {df.index}")
print(f"veri tipi: {df.dtypes}")
