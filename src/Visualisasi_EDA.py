import matplotlib.pyplot as plt
import pandas as pd

#@title Class distribution plot
target.value_counts().plot(kind='bar')
plt.title("Distribusi Kelas Target (Bladder Inflammation)")
plt.xlabel("Kelas")
plt.ylabel("Jumlah Data")
plt.show()

#@title Histogram temperature
plt.hist(features['temperature'], bins=10)
plt.title("Distribusi Suhu Tubuh Pasien (Temperature)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frekuensi")
plt.show()

#@title Crosstab untuk fitur biner vs target
cross_tab = pd.crosstab(
    features['urine-pushing'],
    target
)

# Bar plot
cross_tab.plot(kind='bar')
plt.title("Hubungan Urine Pushing dengan Bladder Inflammation")
plt.xlabel("Urine Pushing")
plt.ylabel("Jumlah Data")
plt.show()

