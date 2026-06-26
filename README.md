# Student Performance Classification (Klasifikasi Kelulusan Siswa)

Proyek _Machine Learning_ ini bertujuan untuk memprediksi probabilitas kelulusan siswa (dengan target rata-rata skor Matematika, Membaca, dan Menulis $\ge$ 70) menggunakan algoritma **Random Forest Classifier**.

Fokus utama dari proyek ini bukan sekadar mencapai akurasi maksimal, melainkan membangun **Early Warning System (Sistem Peringatan Dini)** yang realistis untuk institusi pendidikan dengan menghindari _Data Leakage_.

## Latar Belakang & Pendekatan Model

Jika model dilatih menggunakan skor ujian untuk memprediksi kelulusan, model akan mendapatkan akurasi mendekati 100%. Namun, model tersebut tidak berguna di dunia nyata karena kita tidak membutuhkan AI jika nilai ujian sudah keluar.

Oleh karena itu, pada proyek ini, **fitur skor ujian (Math, Reading, Writing) sengaja dihapus (di-drop) saat proses _training_**.

Model ini dipaksa untuk memprediksi kelulusan murni berdasarkan **faktor sosio-demografis** siswa (seperti gender, tingkat pendidikan orang tua, jenis makan siang, dan keikutsertaan kursus persiapan). Tujuannya adalah agar sekolah dapat mengidentifikasi siswa yang berisiko tidak lulus _sebelum_ ujian berlangsung, sehingga bisa diberikan bimbingan tambahan.

## 📂 Struktur Repositori

```text
student-performance-classification/
│
├── data/
│   └── StudentsPerformance.csv         # Dataset asli
│
├── notebooks/
│   └── 01_eda_and_experimentation.ipynb # Jurnal riset, visualisasi (EDA), dan eksperimen
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Script untuk memuat, membersihkan, dan encoding data
│   └── train_model.py                  # Script utama untuk melatih dan mengevaluasi model
│
├── requirements.txt                    # Daftar library yang dibutuhkan
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Dataset

**StudentsPerformance.csv**

It contains information about:

- Gender
- Race/Ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Math score
- Reading score
- Writing score

---

## 📈 Machine Learning Workflow

1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Prediction

---

## 👨‍💻 Author

**Amay**

```

```
