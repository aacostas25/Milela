import streamlit as st
import pandas as pd
import json, os, glob, unicodedata
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Optional, Iterable
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
# Recomendador avanzado por mito favorito
# ======================

def _ensure_list(x) -> List[str]:
    if x is None: return []
    if isinstance(x, str): return [x]
    return list(x)

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set([s.lower() for s in a]), set([s.lower() for s in b])
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def _topic_overlap(row_topics: List[str], query_topics: List[str]) -> float:
    return _jaccard(row_topics, query_topics)

def _country_match(row_country: str, pref_countries: List[str]) -> float:
    return 1.0 if pref_countries and row_country.lower() in {c.lower() for c in pref_countries} else 0.0

def _apply_filters(cand_df: pd.DataFrame,
                   include_countries: Optional[List[str]]=None,
                   include_topics: Optional[List[str]]=None) -> pd.DataFrame:
    sub = cand_df
    if include_countries:
        s = {c.lower() for c in include_countries}
        sub = sub[sub["pais"].str.lower().isin(s)]
    if include_topics:
        s = {t.lower() for t in include_topics}
        sub = sub[sub["temas_top3"].apply(lambda xs: any(t.lower() in s for t in xs))]
    return sub

def normalize_title(t: str) -> str:
    """Normaliza títulos quitando tildes y mayúsculas para comparar duplicados."""
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("utf-8")
    return t.lower().strip()

def recommend_similar_to_item(item_id: str) -> pd.DataFrame:
    """
    Recomienda mitos similares a uno dado combinando similitud semántica,
    similitud temática y coincidencia de país.
    Parámetros fijos para Streamlit:
        top_k = 8
        fetch_k = 200
        w_sem = 0.75
        w_topic = 0.20
        w_country = 0.05
    """
    # --- parámetros fijos ---
    top_k = 8
    fetch_k = 200
    w_sem = 0.75
    w_topic = 0.20
    w_country = 0.05
    lambda_mmr = 0.6
    country_filter = None
    topic_filter = None

    # --- verificación de ID ---
    if item_id not in ID2ROW:
        raise KeyError(f"id no encontrado: {item_id}")

    row_idx = ID2ROW[item_id]
    qv = emb[row_idx:row_idx+1]
    D, I = index.search(qv, fetch_k + 1)
    sims, idxs = D.ravel(), I.ravel()

    cand = df_artefactos.iloc[idxs].copy()
    cand["sim_sem"] = sims

    # 🔸 título base normalizado
    base_title = normalize_title(df_artefactos.iloc[row_idx]["titulo"])

    # 🔸 excluir el mismo mito y los títulos iguales (incluso de otros países)
    cand = cand[cand.index != row_idx]
    cand = cand[cand["titulo"].apply(lambda t: normalize_title(t) != base_title)]

    # 🔸 filtros (ninguno activo por defecto)
    cand = _apply_filters(cand, include_countries=country_filter, include_topics=topic_filter)
    if cand.empty:
        return pd.DataFrame(columns=["id","pais","region","titulo","temas_top3_str","score","sim_sem"])

    # 🔸 cálculos de similitud
    base_topics = df_artefactos.iloc[row_idx]["temas_top3"]
    pref_countries = _ensure_list(country_filter)
    cand["sim_topic"] = cand["temas_top3"].apply(lambda xs: _topic_overlap(xs, base_topics))
    cand["country_pref"] = cand["pais"].apply(lambda c: _country_match(c, pref_countries))
    cand["score"] = w_sem*cand["sim_sem"] + w_topic*cand["sim_topic"] + w_country*cand["country_pref"]

    # 🔸 eliminar títulos duplicados dentro del resultado
    cand = cand.sort_values("score", ascending=False)
    cand = cand.loc[~cand["titulo"].apply(normalize_title).duplicated(keep="first")]
    out = cand.head(top_k)

    cols = ["id","pais","region","titulo","temas_top3_str","score","sim_sem","sim_topic","country_pref","texto"]
    return out.assign(id=[ROW2ID[i] for i in out.index]).reindex(columns=cols)



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
        "México": ["Coatlicue", "Cegua", "Chaac"],
        "Perú": ["Inkarri", "Catarata Gocta", "Bufeo Colorado"],
        "Uruguay": ["Caipora", "El Currinche", "Cachimba Del Rey"]
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
            match = df_artefactos[df_artefactos["titulo"].str.lower() == mito_favorito.lower()]
            if not match.empty:
                mito_id = match.iloc[0]["id"]
                recomendaciones = recommend_similar_to_item(item_id=mito_id)
            # Buscar el ID correspondiente al título seleccionado
            #match = df_artefactos[df_artefactos["titulo"].str.lower() == mito_favorito.lower()]
            #if not match.empty:
                #mito_id = match.iloc[0]["id"]
                #recomendaciones = recommend_similar_to_item(item_id=mito_id, top_k=5)
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
