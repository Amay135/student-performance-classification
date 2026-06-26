from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Mengimpor fungsi dari file data_preprocessing.py yang ada di folder yang sama
from data_preprocessing import load_and_preprocess_data

def train_and_evaluate():
    print("=== Memulai Proses Training ===")
    
    # 1. Memuat dan memproses data
    # (Pastikan file CSV berada di dalam folder 'data' di luar folder 'src')
    filepath = '../data/StudentsPerformance.csv' 
    print(f"Memuat data dari: {filepath}")
    
    try:
        X, y = load_and_preprocess_data(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} tidak ditemukan. Periksa struktur folder Anda.")
        return

    # 2. Membagi data menjadi Data Latih (80%) dan Data Uji (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Jumlah data latih : {X_train.shape[0]} baris")
    print(f"Jumlah data uji   : {X_test.shape[0]} baris")
    
    # 3. Melatih model Random Forest
    print("\nMelatih model Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Mengevaluasi Model
    print("\n=== Hasil Evaluasi ===")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model : {acc * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    
    # 5. Menyimpan model ke dalam file .pkl
    model_filename = 'random_forest_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModel berhasil disimpan sebagai '{model_filename}'")

if __name__ == "__main__":
    train_and_evaluate()