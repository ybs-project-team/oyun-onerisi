import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- BELLEK DOSTU YÜKLEME ---
@st.cache_data
def load_data():
    df = pd.read_csv('steam.csv')
    df['genres'] = df['genres'].fillna('')
    return df

@st.cache_resource
def get_similarity_matrix(df):
    cv = CountVectorizer()
    genre_matrix = cv.fit_transform(df['genres'])
    return cosine_similarity(genre_matrix, genre_matrix)

df = load_data()
cosine_sim = get_similarity_matrix(df)

# --- ARAYÜZ ---
st.set_page_config(page_title="Oyun Arama Motoru", page_icon="🎮")
st.markdown("<h1 style='text-align: center;'>🎮 Akıllı Oyun Öneri Motoru</h1>", unsafe_allow_html=True)

# Oturum yönetimi (Sayfa her tıklandığında donmasın diye)
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

game_name = st.text_input("Aramak istediğiniz oyunun adını yazın:", value=st.session_state.search_query)

if game_name:
    match = df[df['name'].str.lower() == game_name.lower()]
    if not match.empty:
        idx = match.index[0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
        
        st.success(f"'{match.iloc[0]['name']}' oyununa benzeyenler:")
        for i, score in sim_scores:
            st.write(f"🔹 **{df['name'].iloc[i]}**")
    else:
        st.warning("Tam eşleşme yok, şunlara bakabilirsin:")
        partial = df[df['name'].str.contains(game_name, case=False, na=False)]['name'].head(5)
        for name in partial:
            if st.button(name):
                st.session_state.search_query = name
                st.rerun()

st.caption(f"Veritabanı: {len(df)} oyun")