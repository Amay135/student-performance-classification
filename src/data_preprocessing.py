import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    """
    Fungsi untuk memuat dan memproses dataset Students Performance.
    """
    # 1. Memuat data
    df = pd.read_csv(filepath)
    
    # 2. Membuat kolom target (Rata-rata >= 70 dianggap Lulus)
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
    df['lulus'] = (df['average_score'] >= 70).astype(int)
    
    # 3. Memisahkan fitur (X) dan target (y)
    # Kolom skor dan rata-rata dibuang agar model memprediksi 
    # hanya berdasarkan fitur demografis (gender, lunch, dll)
    X = df.drop(['math score', 'reading score', 'writing score', 'average_score', 'lulus'], axis=1)
    y = df['lulus']
    
    # 4. Encoding data kategorikal (Label Encoding)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])
        
    return X, y

if __name__ == "__main__":
    # Blok ini hanya berjalan jika file ini dieksekusi langsung 
    # Berguna untuk melakukan pengecekan/testing
    X, y = load_and_preprocess_data('../data/StudentsPerformance.csv')
    print("Preprocessing berhasil!")
    print(f"Ukuran Data Fitur (X): {X.shape}")