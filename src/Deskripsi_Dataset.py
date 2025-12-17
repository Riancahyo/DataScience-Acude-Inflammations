#@title Target prediksi

print("Daftar target tersedia:", list(targets.columns))
print("Target yang diprediksi:", list(targets.columns)[0])

# Simpan target
target = targets.iloc[:, 0]

#@title Jumlah kolom fitur
features.shape

#@title Jumlah kolom target
targets.shape

#@title Isi kolom fitur dan target
print("Isi kolom fitur dan target:")
features.info()
target.info()

#@title Tipe data kolom fitur
features.dtypes

#@title Tipe data kolom target
target.dtypes
target.info()

display(features)
display(target)

