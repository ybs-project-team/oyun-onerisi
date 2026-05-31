import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    df = pd.read_csv('steam.csv')
    df['genres'] = df['genres'].fillna('')
    # Sadece lazım olan sütunları tutarak bellek tasarrufu yapıyoruz
    return df[['name', 'genres']]

df = load_data()

# --- ARAYÜZ ---
st.set_page_config(page_title="Oyun Arama Motoru", page_icon="🎮")
st.markdown("<h1 style='text-align: center;'>🎮 Akıllı Oyun Öneri Motoru</h1>", unsafe_allow_html=True)

# Oturum yönetimi
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

game_name = st.text_input("Aramak istediğiniz oyunun adını yazın:", value=st.session_state.search_query)

if game_name:
    match = df[df['name'].str.lower() == game_name.lower()]
    
    if not match.empty:
        idx = match.index[0]
        
        # BELLEK DOSTU İŞLEM: Matrisin tamamını değil, sadece gerektiğinde hesapla
        cv = CountVectorizer()
        genre_matrix = cv.fit_transform(df['genres'])
        
        # Sadece aranan oyunun benzerlik skorlarını hesapla
        sim_scores = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()
        
        # En benzer 5 oyunu getir
        related_indices = sim_scores.argsort()[-6:-1][::-1]
        
        st.success(f"'{match.iloc[0]['name']}' oyununa benzeyenler:")
        for i in related_indices:
            st.write(f"🔹 **{df['name'].iloc[i]}**")
    else:
        st.warning("Tam eşleşme bulunamadı. İşte benzer isimli oyunlar:")
        partial = df[df['name'].str.contains(game_name, case=False, na=False)]['name'].head(5)
        for name in partial:
            if st.button(name):
                st.session_state.search_query = name
                st.rerun()

st.caption(f"Veritabanı: {len(df)} oyun")