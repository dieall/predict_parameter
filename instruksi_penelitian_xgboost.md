# Instruksi Penelitian Prediksi Parameter Menggunakan XGBoost

## 1. Lokasi File Notebook

Buatlah notebook prediksi dengan nama file berikut:

```text
C:\Aldi\#KULIAH\#BISMILLAH YA ALLAH YA TUHANKU SKRIPSI AN\prediksi.ipynb
```

## 2. Lokasi Dataset

Dataset yang digunakan adalah file Excel berikut:

```text
C:\Aldi\#KULIAH\#BISMILLAH YA ALLAH YA TUHANKU SKRIPSI AN\Data_Chemical_2026.04-16-MT620.xlsx
```

## 3. Tujuan Penelitian

Penelitian ini bertujuan untuk membuat model prediksi hasil analisis setiap parameter menggunakan algoritma **Extreme Gradient Boosting (XGBoost)**. Parameter yang diprediksi meliputi setiap parameter hasil analisis, mulai dari **Transmission** hingga **PT**.

Setiap parameter akan diproses secara terpisah. Artinya, setiap parameter dijadikan sebagai target prediksi masing-masing, kemudian dilakukan proses pemodelan menggunakan XGBoost. Dataset akan dibagi menjadi data latih dan data uji melalui proses **data splitting** agar performa model dapat dievaluasi secara objektif terhadap data yang belum pernah dilihat oleh model.

Setiap tahapan penelitian, mulai dari pengolahan data awal hingga evaluasi model, harus disertai dengan penjelasan metode, alasan penggunaan metode, serta rujukan jurnal yang relevan. Rujukan jurnal digunakan untuk mendukung dasar ilmiah dari proses preprocessing, pembagian data, pemilihan algoritma XGBoost, evaluasi model, dan interpretasi hasil.

---

# Tahapan Penelitian

## 1. Identifikasi Masalah

Tahap pertama adalah mengidentifikasi permasalahan penelitian, yaitu bagaimana membangun model machine learning yang mampu memprediksi hasil analisis parameter kimia berdasarkan data historis produksi atau data hasil pengujian sebelumnya.

Pada tahap ini, ditentukan bahwa algoritma yang digunakan adalah **XGBoost Regression**, karena data yang diprediksi berupa nilai numerik atau hasil analisis parameter.

### Alasan Penggunaan Metode

- XGBoost mampu menangani hubungan data yang kompleks dan non-linear.
- XGBoost sering digunakan untuk masalah prediksi berbasis data historis.
- XGBoost memiliki performa tinggi pada data tabular.

### Rujukan Jurnal

Chen dan Guestrin menjelaskan bahwa XGBoost merupakan sistem tree boosting yang scalable dan efektif untuk berbagai masalah prediksi machine learning.

- Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System**. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
- Link: https://dl.acm.org/doi/10.1145/2939672.2939785
- Alternatif PDF/arXiv: https://arxiv.org/abs/1603.02754

---

## 2. Pengumpulan Data

Data penelitian diambil dari file Excel:

```text
Data_Chemical_2026.04-16-MT620.xlsx
```

Dataset ini berisi data historis hasil analisis parameter kimia. Data tersebut digunakan sebagai dasar pembentukan model prediksi.

Pada tahap ini dilakukan pembacaan dataset menggunakan Python, misalnya dengan library `pandas`.

### Contoh Kode

```python
import pandas as pd

file_path = r"C:\Aldi\#KULIAH\#BISMILLAH YA ALLAH YA TUHANKU SKRIPSI AN\Data_Chemical_2026.04-16-MT620.xlsx"
df = pd.read_excel(file_path)
```

### Alasan Penggunaan Metode

- Data historis diperlukan agar model machine learning dapat mempelajari pola dari data sebelumnya.
- Format Excel umum digunakan dalam pengolahan data laboratorium atau data produksi.

### Rujukan Jurnal

Penelitian prediksi menggunakan XGBoost pada data historis juga dilakukan dalam studi prediksi permintaan kalibrasi, di mana data historis digunakan sebagai dasar pembentukan model prediksi.

- **Algoritma XGBoost untuk Analisis Prediksi Permintaan Kalibrasi di PT PAL Indonesia**
- Link: https://ejournal.akakom.ac.id/index.php/jiko/article/download/2245/pdf

---

## 3. Eksplorasi Data Awal

Tahap ini dilakukan untuk memahami struktur dataset.

Analisis awal meliputi:

1. Melihat jumlah baris dan kolom.
2. Melihat nama-nama kolom.
3. Melihat tipe data setiap kolom.
4. Mengecek nilai kosong atau missing value.
5. Mengecek data duplikat.
6. Melihat statistik deskriptif seperti mean, minimum, maksimum, dan standar deviasi.

### Contoh Kode

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### Alasan Penggunaan Metode

- Eksplorasi data diperlukan agar peneliti memahami kondisi dataset sebelum melakukan pemodelan.
- Tahap ini membantu menentukan strategi preprocessing yang tepat.

---

## 4. Preprocessing Data

Preprocessing dilakukan untuk memastikan data siap digunakan oleh model XGBoost.

Tahapan preprocessing dapat meliputi:

1. Menghapus data duplikat.
2. Menangani missing value.
3. Mengubah format tanggal jika terdapat kolom waktu.
4. Mengubah data kategorikal menjadi numerik jika diperlukan.
5. Menghapus kolom yang tidak relevan.
6. Memastikan seluruh fitur yang digunakan memiliki tipe data numerik.
7. Menangani outlier apabila diperlukan.

### Contoh Kode

```python
df = df.drop_duplicates()
df = df.dropna()
```

Jika ada kolom kategorikal:

```python
df = pd.get_dummies(df, drop_first=True)
```

### Alasan Penggunaan Metode

- Data yang tidak bersih dapat menurunkan akurasi model.
- XGBoost dapat menangani data numerik dengan baik, tetapi data kategorikal perlu disesuaikan terlebih dahulu jika belum dalam format numerik.
- Preprocessing membantu meningkatkan kualitas input model.

### Rujukan Jurnal

Studi mengenai pengaruh preprocessing terhadap performa XGBoost menunjukkan bahwa teknik preprocessing dapat memengaruhi hasil prediksi model.

- **Impact of Data Pre-Processing Techniques on XGBoost Model Performance**
- Link: https://www.mdpi.com/2673-7426/4/4/118

---

## 5. Penentuan Fitur dan Target

Pada tahap ini, setiap parameter dari **Transmission** hingga **PT** dijadikan sebagai target prediksi secara bergantian.

Contoh target:

- Target 1: Transmission
- Target 2: Parameter setelah Transmission
- Target 3: Parameter berikutnya
- Target terakhir: PT

Untuk setiap target, kolom parameter tersebut dijadikan sebagai variabel `y`, sedangkan kolom lainnya dijadikan sebagai fitur `X`.

### Contoh Kode

```python
target = "Transmission"

X = df.drop(columns=[target])
y = df[target]
```

Jika parameter target banyak, proses dapat dibuat menggunakan perulangan:

```python
target_columns = ["Transmission", "...", "PT"]

for target in target_columns:
    X = df.drop(columns=[target])
    y = df[target]
```

### Alasan Penggunaan Metode

- Setiap parameter dianalisis secara terpisah agar model dapat mempelajari pola khusus untuk masing-masing parameter.
- Pendekatan ini memungkinkan hasil evaluasi setiap parameter dibandingkan secara individual.

---

## 6. Pembagian Data Training dan Testing

Dataset dibagi menjadi data latih dan data uji. Data latih digunakan untuk melatih model, sedangkan data uji digunakan untuk mengevaluasi performa model.

Rasio pembagian data yang dapat digunakan:

```text
80% data training
20% data testing
```

### Contoh Kode

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Jika data memiliki urutan waktu atau bersifat time series, maka split sebaiknya tidak dilakukan secara acak. Data lama digunakan sebagai training, sedangkan data terbaru digunakan sebagai testing.

### Alasan Penggunaan Metode

- Train-test split digunakan untuk menguji apakah model mampu memprediksi data baru.
- Pembagian data mencegah evaluasi yang bias karena model diuji pada data yang tidak digunakan saat pelatihan.
- Jika data berbasis waktu, time-series split lebih sesuai karena model seharusnya belajar dari masa lalu untuk memprediksi data masa depan.

### Rujukan Jurnal

Rácz dan Héberger membahas bahwa rasio train/test split dapat memengaruhi performa model machine learning.

- Rácz, A., & Héberger, K. **Effect of Dataset Size and Train/Test Split Ratios in QSAR/QSPR Multiclass Classification**.
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC7922354/

Penelitian perbandingan Random Forest dan XGBoost untuk prediksi berbasis data historis juga menggunakan pendekatan data latih dari periode sebelumnya dan data uji dari periode terbaru untuk mencerminkan kondisi prediksi nyata.

- **Perbandingan Random Forest dan XGBoost untuk Prediksi Penjualan**
- Link: https://ejurnal.seminar-id.com/index.php/bits/article/download/8491/4230/

---

## 7. Pembuatan Model XGBoost

Model yang digunakan adalah **XGBoost Regressor**, karena target yang diprediksi berupa nilai numerik.

### Contoh Kode

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

### Alasan Penggunaan Metode

- XGBoost merupakan pengembangan dari gradient boosting.
- Model ini membangun banyak decision tree secara bertahap.
- Setiap tree baru berusaha memperbaiki kesalahan dari tree sebelumnya.
- XGBoost efektif untuk data tabular, data non-linear, dan data dengan banyak fitur.

### Rujukan Jurnal

Chen dan Guestrin memperkenalkan XGBoost sebagai algoritma tree boosting yang efisien, scalable, dan banyak digunakan dalam machine learning.

- Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System**.
- Link: https://arxiv.org/abs/1603.02754

Penelitian prediksi menggunakan XGBoost di Indonesia juga menunjukkan bahwa XGBoost cocok digunakan untuk data non-linear dan mampu meningkatkan akurasi prediksi.

- **Algoritma XGBoost untuk Analisis Prediksi Permintaan Kalibrasi di PT PAL Indonesia**
- Link: https://ejournal.akakom.ac.id/index.php/jiko/article/download/2245/pdf

---

## 8. Training Model

Pada tahap ini, model XGBoost dilatih menggunakan data training.

### Contoh Kode

```python
model.fit(X_train, y_train)
```

Model akan mempelajari hubungan antara fitur input dan target parameter yang sedang diprediksi.

### Alasan Penggunaan Metode

- Training diperlukan agar model mengenali pola historis.
- Model belajar dari kombinasi fitur yang tersedia untuk menghasilkan prediksi terhadap target.

---

## 9. Prediksi Hasil Analisis

Setelah model dilatih, model digunakan untuk memprediksi data testing.

### Contoh Kode

```python
y_pred = model.predict(X_test)
```

Hasil prediksi kemudian dibandingkan dengan nilai aktual dari data testing.

```python
result = pd.DataFrame({
    "Actual": y_test,
    "Prediction": y_pred
})
```

### Alasan Penggunaan Metode

- Perbandingan antara nilai aktual dan nilai prediksi diperlukan untuk mengetahui seberapa dekat hasil prediksi model terhadap data sebenarnya.

---

## 10. Evaluasi Model

Evaluasi model dilakukan menggunakan metrik regresi, seperti:

1. **MAE** atau Mean Absolute Error.
2. **MSE** atau Mean Squared Error.
3. **RMSE** atau Root Mean Squared Error.
4. **R² Score**.

### Contoh Kode

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

### Alasan Penggunaan Metode

- MAE menunjukkan rata-rata kesalahan absolut prediksi.
- RMSE memberikan penalti lebih besar terhadap kesalahan prediksi yang besar.
- R² menunjukkan seberapa baik model menjelaskan variasi data target.
- Kombinasi beberapa metrik membuat evaluasi model lebih lengkap.

### Rujukan Jurnal

Penelitian prediksi harga saham menggunakan XGBoost dan metode pembanding menggunakan MAE dan RMSE sebagai metrik evaluasi performa model.

- **Forecasting Indonesian Banking Stock Prices Using Prophet, XGBoost, and Ridge Regression**
- Link: https://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/4973

Penelitian prediksi unemployment rate menggunakan XGBoost juga mengevaluasi performa model dengan MSE, RMSE, dan MAPE.

- **Unemployment Rate Prediction System Using Deep Learning and XGBoost**
- Link: https://internetworkingindonesia.org/index.php/iij/article/view/160

---

## 11. Visualisasi Hasil Prediksi

Visualisasi dilakukan untuk membandingkan nilai aktual dan nilai prediksi.

### Contoh Kode

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Prediction")
plt.legend()
plt.title(f"Actual vs Prediction - {target}")
plt.show()
```

Bentuk visualisasi yang dapat digunakan:

1. Grafik aktual vs prediksi.
2. Scatter plot aktual dan prediksi.
3. Grafik error atau residual.
4. Feature importance dari XGBoost.

Contoh feature importance:

```python
import xgboost as xgb

xgb.plot_importance(model)
plt.show()
```

### Alasan Penggunaan Metode

- Visualisasi membantu melihat pola kesalahan prediksi.
- Feature importance membantu mengetahui variabel mana yang paling berpengaruh terhadap hasil prediksi.

### Rujukan Jurnal

Penelitian mengenai performa XGBoost dalam prediksi harga saham menggunakan feature importance untuk menentukan fitur yang paling optimal dalam prediksi.

- **Performance Analysis of XGBoost Algorithm to Determine the Most Optimal Parameters and Features in Predicting Stock Price Movement**
- Link: https://jurnal.upnyk.ac.id/index.php/telematika/article/view/9329

---

## 12. Interpretasi Hasil

Pada tahap ini, hasil evaluasi setiap parameter dijelaskan.

Interpretasi dapat mencakup:

1. Parameter mana yang memiliki hasil prediksi paling baik.
2. Parameter mana yang memiliki error paling besar.
3. Nilai MAE, RMSE, dan R² untuk setiap parameter.
4. Fitur yang paling berpengaruh terhadap setiap parameter.
5. Penyebab kemungkinan error prediksi.

### Contoh Narasi

> Berdasarkan hasil evaluasi, parameter Transmission memperoleh nilai RMSE yang lebih rendah dibandingkan parameter lainnya. Hal ini menunjukkan bahwa model XGBoost mampu mempelajari pola data Transmission dengan lebih baik. Sementara itu, parameter tertentu dengan nilai RMSE tinggi menunjukkan bahwa pola datanya lebih kompleks atau memiliki variasi yang lebih besar.

---

## 13. Kesimpulan

Tahap akhir adalah menyusun kesimpulan berdasarkan hasil penelitian.

Kesimpulan dapat berisi:

1. XGBoost dapat digunakan untuk memprediksi hasil analisis setiap parameter.
2. Setiap parameter memiliki tingkat akurasi yang berbeda.
3. Parameter dengan nilai error rendah menunjukkan bahwa pola datanya lebih mudah dipelajari oleh model.
4. Parameter dengan nilai error tinggi membutuhkan evaluasi lanjutan, seperti penambahan data, tuning hyperparameter, atau pemilihan fitur yang lebih baik.
5. Model dapat digunakan sebagai pendekatan awal untuk membantu prediksi hasil analisis berbasis data historis.

---

# Ringkasan Tahapan Notebook

Notebook penelitian harus memuat tahapan sebagai berikut:

1. Identifikasi masalah penelitian.
2. Pengumpulan dataset dari file Excel.
3. Eksplorasi data awal.
4. Preprocessing data.
5. Penentuan fitur dan target untuk setiap parameter dari Transmission hingga PT.
6. Pembagian data training dan testing.
7. Pembuatan model XGBoost Regressor.
8. Training model untuk setiap parameter.
9. Prediksi nilai setiap parameter.
10. Evaluasi model menggunakan MAE, MSE, RMSE, dan R².
11. Visualisasi hasil aktual dan prediksi.
12. Analisis feature importance.
13. Interpretasi hasil.
14. Kesimpulan.

---

# Daftar Rujukan Jurnal

| No | Bagian Metode | Rujukan | Link | Alasan Digunakan |
|---|---|---|---|---|
| 1 | Algoritma XGBoost | Chen & Guestrin, **XGBoost: A Scalable Tree Boosting System** | https://dl.acm.org/doi/10.1145/2939672.2939785 | Dasar utama algoritma XGBoost. |
| 2 | Algoritma XGBoost | Chen & Guestrin, arXiv version | https://arxiv.org/abs/1603.02754 | Alternatif akses artikel XGBoost. |
| 3 | XGBoost untuk prediksi | **Algoritma XGBoost untuk Analisis Prediksi Permintaan Kalibrasi di PT PAL Indonesia** | https://ejournal.akakom.ac.id/index.php/jiko/article/download/2245/pdf | Contoh penerapan XGBoost untuk prediksi pada data historis. |
| 4 | Preprocessing | **Impact of Data Pre-Processing Techniques on XGBoost Model Performance** | https://www.mdpi.com/2673-7426/4/4/118 | Menjelaskan dampak preprocessing pada performa XGBoost. |
| 5 | Data splitting | Rácz & Héberger, **Effect of Dataset Size and Train/Test Split Ratios** | https://pmc.ncbi.nlm.nih.gov/articles/PMC7922354/ | Dasar penggunaan pembagian data training dan testing. |
| 6 | Time-series split | **Perbandingan Random Forest dan XGBoost untuk Prediksi Penjualan** | https://ejurnal.seminar-id.com/index.php/bits/article/download/8491/4230/ | Rujukan jika data memiliki urutan waktu dan perlu split berdasarkan periode. |
| 7 | Evaluasi model | **Forecasting Indonesian Banking Stock Prices Using Prophet, XGBoost, and Ridge Regression** | https://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/4973 | Rujukan penggunaan MAE dan RMSE untuk evaluasi model prediksi. |
| 8 | Evaluasi model | **Unemployment Rate Prediction System Using Deep Learning and XGBoost** | https://internetworkingindonesia.org/index.php/iij/article/view/160 | Rujukan penggunaan MSE, RMSE, dan MAPE pada evaluasi XGBoost. |
| 9 | Feature importance | **Performance Analysis of XGBoost Algorithm to Determine the Most Optimal Parameters and Features in Predicting Stock Price Movement** | https://jurnal.upnyk.ac.id/index.php/telematika/article/view/9329 | Rujukan penggunaan feature importance dalam XGBoost. |
| 10 | XGBoost untuk data kualitas | **Algoritma XGBoost untuk Klasifikasi Kualitas Air Minum** | https://ejournal.itn.ac.id/jati/article/view/7308 | Rujukan tambahan bahwa XGBoost digunakan pada data parameter kualitas. |

---

# Catatan Tambahan

Jika data memiliki kolom waktu atau tanggal, sebaiknya gunakan pembagian data berdasarkan urutan waktu, bukan pembagian acak. Hal ini bertujuan agar model dilatih menggunakan data masa lalu dan diuji menggunakan data yang lebih baru, sehingga lebih sesuai dengan kondisi prediksi nyata.

Jika data tidak memiliki urutan waktu yang jelas, maka pembagian acak menggunakan `train_test_split` dengan rasio 80:20 dapat digunakan sebagai pendekatan awal.
