import streamlit as st
import pandas as pd
import json, os, glob, unicodedata
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ======================
# Carga de datos
# ======================
@st.cache_data
def load_country_jsons(data_dir):
    rows = []
    json_paths = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No se encontraron JSON en: {data_dir}")
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
            for it in arr:
                rows.append({
                    "id": str(it.get("id","")).strip(),
                    "titulo": str(it.get("titulo","")).strip(),
                    "pais": str(it.get("pais","")).strip(),
                    "region": str(it.get("region","")).strip(),
                    "texto": str(it.get("texto","")).strip()
                })
    df = pd.DataFrame(rows).dropna(subset=["texto"])
    df = df[df["texto"].str.len() > 20].reset_index(drop=True)
    return df

DATA_DIR = "InfoCompleta"
df = load_country_jsons(DATA_DIR)

# ======================
# Carga artefactos y modelo
# ======================
@st.cache_resource
def load_artifacts():
    df_artefactos = pd.read_json("milela_enriquecido.json", encoding="utf-8")
    index = faiss.read_index("milela_faiss.index")
    emb = np.load("milela_embeddings.npy")
    return df_artefactos, index, emb

df_artefactos, index, emb = load_artifacts()

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

sbert = load_sbert_model()

# ======================
# Función de búsqueda textual
# ======================
def buscar_mitos_por_texto(query, top_k=5):
    qv = sbert.encode([query], convert_to_numpy=True).astype("float32")
    qv = normalize(qv, norm="l2", axis=1)
    D, I = index.search(qv, top_k)
    resultados = df_artefactos.iloc[I[0]].copy()
    resultados["score"] = D[0]
    return resultados[["id","pais", "region", "titulo", "temas_top3_str", "score", "texto"]]

# ======================
# Recomendador por mito favorito
# ======================
def normalize_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("utf-8")
    return t.lower().strip()

ID2ROW = {str(row["id"]): i for i, row in df_artefactos.reset_index().iterrows()}
ROW2ID = {i: str(row["id"]) for i, row in df_artefactos.reset_index().iterrows()}

def recommend_similar_to_item(item_title: str, top_k: int = 5):
    """Busca en el dataset el mito más cercano por título y recomienda otros similares."""
    # Buscar la fila cuyo título coincida (tolerante a mayúsculas)
    mask = df_artefactos["titulo"].str.lower() == item_title.lower()
    if not mask.any():
        return pd.DataFrame(columns=["pais","region","titulo","temas_top3_str","score","texto"])
    
    row_idx = mask.idxmax()
    base_title = normalize_title(df_artefactos.loc[row_idx, "titulo"])
    qv = emb[row_idx:row_idx+1]
    D, I = index.search(qv, 200)

    cand = df_artefactos.iloc[I[0]].copy()
    cand["sim_sem"] = D[0]
    cand = cand[cand.index != row_idx]
    cand = cand[cand["titulo"].apply(lambda t: normalize_title(t) != base_title)]
    cand = cand.sort_values("sim_sem", ascending=False).head(top_k)

    return cand[["id","pais","region","titulo","temas_top3_str","sim_sem","texto"]]

# ======================
# Interfaz principal
# ======================
st.title("🌎✨ MILELA – Mitos y Leyendas de Latinoamérica")
st.write("""
**Milela** integra, analiza y recomienda mitos y leyendas latinoamericanos
usando técnicas de **procesamiento del lenguaje natural (NLP)**.
Permite explorar, buscar y descubrir historias de toda la región.
""")

tabs = st.tabs([
    "🔍 Buscar por temática",
    "📖 Explorar mitos por país",
    "📋 Encuesta de preferencias"
])

# --- TAB 3: Encuesta ---
with tabs[2]:
    st.subheader("Encuesta de preferencias")

    nombre = st.text_input("Nombre")
    edad = st.number_input("Edad", min_value=5, max_value=120, step=1)
    pais = st.selectbox("País de origen", 
                        ["Argentina","Bolivia","Chile","Colombia","Ecuador","México","Perú","Uruguay"])

    mitos_por_pais = {
        "Argentina": ["El Familiar", "Luz Mala", "Pombero"],
        "Bolivia": ["Jichi", "Ekeko", "Chancho Verde"],
        "Chile": ["El Caleuche", "La Prodigiosa Pincoya", "El Trauco"],
        "Colombia": ["Bolefuego", "Madremonte", "Patasola"],
        "Ecuador": ["Bambero", "La Dama Tapada", "El Riviel"],
        "México": ["La Nahuala", "El Chupacabras", "La Llorona"],
        "Perú": ["El Tunche", "La Jarjacha", "El Pishtaco"],
        "Uruguay": ["El Lobizón", "La Luz Mala", "El Pombero"]
    }

    mito_favorito = st.selectbox("Mito o leyenda favorita", mitos_por_pais[pais])

    if st.button("Enviar"):
        if nombre:
            st.success(f"Gracias {nombre}, tus datos fueron registrados.")
            st.write(f"""
            **Resumen de tus respuestas:**
            - Edad: {edad}
            - País: {pais}
            - Mito favorito: {mito_favorito}
            """)
            # 🔹 Mostrar recomendaciones
            st.divider()
            st.subheader("✨ Recomendaciones similares a tu mito favorito:")

            # Buscar el ID correspondiente al título seleccionado
            match = df_artefactos[df_artefactos["titulo"].str.lower() == mito_favorito.lower()]
            st.write(f"El match fue: {match}")
            if not match.empty:
                mito_id = match.iloc[0]["id"]
                recomendaciones = recommend_similar_to_item(item_id=mito_id, top_k=5)
            else:
                recomendaciones = pd.DataFrame()
        
            if recomendaciones.empty:
                st.info("No se encontraron mitos similares en la base de datos.")
            else:
                for _, row in recomendaciones.iterrows():
                    with st.expander(f"📜 {row['titulo']} ({row['pais']}) – Similitud: {row['sim_sem']:.3f}"):
                        st.write(f"**Temas:** {row['temas_top3_str']}")
                        st.write(f"**Región:** {row['region']}")
                        st.write(row['texto'])

        else:
            st.warning("Por favor, ingresa tu nombre antes de enviar.")

# --- TAB 2: Exploración ---
with tabs[1]:
    st.subheader("Explora los mitos y leyendas por país")
    pais_explorar = st.selectbox(
        "Selecciona un país para explorar sus mitos",
        sorted(df["pais"].unique())
    )
    df_filtrado = df[df["pais"] == pais_explorar]
    if not df_filtrado.empty:
        for _, row in df_filtrado.iterrows():
            with st.expander(f"📜 {row['titulo']}"):
                st.write(f"**Región:** {row['region']}")
                st.write(row["texto"])
    else:
        st.info("No hay mitos disponibles para este país.")

# --- TAB 1: Buscador ---
with tabs[0]:
    st.subheader("Buscar mitos por temática o descripción")
    query = st.text_input("Escribe una palabra o tema (ej: 'espíritus', 'agua', 'rituales')")
    query = query.lower()
    if st.button("Buscar"):
        if query.strip():
            resultados = buscar_mitos_por_texto(query, top_k=5)
            if resultados.empty:
                st.warning("No se encontraron mitos relacionados.")
            else:
                st.success(f"Se encontraron {len(resultados)} mitos relacionados:")
                for _, row in resultados.iterrows():
                    with st.expander(f"📜 {row['titulo']} ({row['pais']}) – Score: {row['score']:.3f}"):
                        st.write(f"**Temas:** {row['temas_top3_str']}")
                        st.write(f"**Región:** {row['region']}")
                        st.write(row['texto'])
        else:
            st.warning("Por favor, escribe un tema o palabra clave.")

st.markdown("---")
st.caption("Proyecto desarrollado por **Andrea Acosta y Alexandra Moraga** – Pontificia Universidad Católica de Chile, 2025")
