import streamlit as st
import pandas as pd
import json, os, glob
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

def buscar_mitos_por_texto(query, top_k=5):
    qv = sbert.encode([query], convert_to_numpy=True).astype("float32")
    qv = normalize(qv, norm="l2", axis=1)
    D, I = index.search(qv, top_k)
    resultados = df_artefactos.iloc[I[0]].copy()
    resultados["score"] = D[0]
    return resultados[["pais", "region", "titulo", "temas_top3_str", "score", "texto"]]

# ======================
# Interfaz principal
# ======================

st.title("🌎✨ MILELA – Mitos y Leyendas de Latinoamérica")
st.write("""
**Milela** integra, analiza y recomienda mitos y leyendas latinoamericanos
usando técnicas de **procesamiento del lenguaje natural (NLP)**.
Permite explorar, buscar y descubrir historias de toda la región.
""")

# === Tabs principales ===
tabs = st.tabs([
    "📋 Encuesta de preferencias",
    "📖 Explorar mitos por país",
    "🔍 Buscar por temática"
])

# --- TAB 1: Encuesta ---
with tabs[2]:
    st.subheader("Encuesta de preferencias")

    nombre = st.text_input("Nombre")
    edad = st.number_input("Edad", min_value=5, max_value=120, step=1)
    pais = st.selectbox("País de origen", 
                        ["Argentina","Bolivia","Chile","Colombia","Ecuador","México","Perú","Uruguay"])

    mitos_por_pais = {
        "Argentina": ["El Familiar", "La Luz Mala", "El Pombero"],
        "Bolivia": ["La Kantuta", "El Ekeko", "La Viuda del Monte"],
        "Chile": ["El Caleuche", "La Pincoya", "El Trauco"],
        "Colombia": ["La Llorona", "Madremonte", "La Patasola"],
        "Ecuador": ["El Duende", "La Dama Tapada", "La Tunda"],
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

# --- TAB 3: Buscador ---
with tabs[0]:
    st.subheader("Buscar mitos por temática o descripción")
    query = st.text_input("Escribe una palabra o tema (ej: 'espíritus', 'agua', 'rituales')")
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
