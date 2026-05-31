import pandas as pd
import numpy as np

print("Tebrikler Tarık! Sistem sorunsuz çalışıyor.")
print("Pandas ve NumPy başarıyla yüklendi.")

import pandas as pd

# Veriyi okuyalım
df = pd.read_csv('steam.csv')

# İlk 5 satırı yazdırıp veri setine bir göz atalım
print("Veri başarıyla okundu!")
print(df.head())

# Oyun isimlerini ve türlerini (genres) görmek için:
print(df[['name', 'genres']].head())

# Türleri bir liste gibi düşünelim (Action;Adventure gibi)
# Birbirine benzer olanları bulmak için basit bir işlem
df['genres'] = df['genres'].str.split(';')
print("Türler artık liste formatında:")
print(df[['name', 'genres']].head())

from sklearn.feature_extraction.text import CountVectorizer

# Listeleri tekrar string'e çeviriyoruz (çünkü CountVectorizer bunu istiyor)
df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))

# Türleri sayısallaştıralım
cv = CountVectorizer()
genre_matrix = cv.fit_transform(df['genres_str'])

print("Veri vektörlere dönüştürüldü, şekli:", genre_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

# 1. Benzerlik matrisini hesapla
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# 2. Öneri fonksiyonunu tanımla
def get_recommendations(game_title, cosine_sim=cosine_sim):
    # Oyun isminin veri setindeki indeksini bul
    try:
        idx = df[df['name'] == game_title].index[0]
    except IndexError:
        return f"'{game_title}' adlı oyun veri setinde bulunamadı!"

    # Seçilen oyunun diğer tüm oyunlarla olan benzerlik skorlarını al
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Skorlara göre büyükten küçüğe sırala
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # En benzer 5 oyunu al (kendisi hariç)
    sim_scores = sim_scores[1:6]
    
    # Oyun indekslerini al ve isimlerini döndür
    game_indices = [i[0] for i in sim_scores]
    return df['name'].iloc[game_indices]

# 3. Test edelim (Veri setindeki bir oyun ismini buraya yaz)
print("\n--- Öneriler ---")
print(get_recommendations('Counter-Strike'))