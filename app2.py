import streamlit as st
import pandas as pd
import json, os, glob, unicodedata
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Optional, Iterable
from groq import Groq  # 🔹 NUEVO

# ======================
# Configuración de Groq
# ======================
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False
    st.warning("⚠️ No se pudo conectar a Groq. Agrega tu API key en secrets.")

def consultar_groq(pregunta: str, contexto_mito: str = "", modelo: str = "mixtral-8x7b-32768") -> str:
    """
    Envía una pregunta a Groq, opcionalmente con contexto de un mito.
    """
    if not GROQ_AVAILABLE:
        return "Error: Groq no está configurado. Agrega tu API key."
    
    if contexto_mito:
        prompt = f"""Eres un experto en mitología y leyendas latinoamericanas.

**Contexto del mito:**
{contexto_mito}

**Pregunta del usuario:**
{pregunta}

Responde de forma clara, informativa y concisa, basándote en el contexto proporcionado."""
    else:
        prompt = f"""Eres un experto en mitología y leyendas latinoamericanas.

**Pregunta:**
{pregunta}

Responde de forma clara, informativa y concisa."""
    
    try:
        completion = groq_client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al consultar Groq: {str(e)}"

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
ID2ROW = {str(row["id"]): i for i, row in df_artefactos.reset_index().iterrows()}
ROW2ID = {i: str(row["id"]) for i, row in df_artefactos.reset_index().iterrows()}

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
# Recomendador avanzado
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
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("utf-8")
    return t.lower().strip()

def recommend_similar_to_item(item_id: str, top_k = 5) -> pd.DataFrame:
    fetch_k = 200
    w_sem = 0.7
    w_topic = 0.25
    w_country = 0.05

    if item_id not in ID2ROW:
        raise KeyError(f"id no encontrado: {item_id}")

    row_idx = ID2ROW[item_id]
    base_row = df_artefactos.iloc[row_idx]
    base_title = normalize_title(base_row["titulo"])
    base_topics = base_row["temas_top3"]

    query_text = f"{base_row['texto']} Temas: {', '.join(base_topics)}"
    qv = sbert.encode([query_text], convert_to_numpy=True).astype("float32")
    qv = normalize(qv, norm="l2", axis=1)

    D, I = index.search(qv, fetch_k + 1)
    sims, idxs = D.ravel(), I.ravel()
    cand = df_artefactos.iloc[idxs].copy()
    cand["sim_sem"] = sims

    cand = cand[cand.index != row_idx]
    cand = cand[cand["titulo"].apply(lambda t: normalize_title(t) != base_title)]

    cand["sim_topic"] = cand["temas_top3"].apply(lambda xs: _jaccard(xs, base_topics))
    cand["score"] = w_sem*cand["sim_sem"] + w_topic*cand["sim_topic"]

    cand = cand.sort_values("score", ascending=False).head(top_k)
    cand = cand.loc[~cand["titulo"].apply(normalize_title).duplicated(keep="first")]

    return cand[["id","pais","region","titulo","temas_top3_str","score","sim_sem","sim_topic","texto"]]

# ======================
# Interfaz principal
# ======================
st.title("🌎✨ MILELA – Mitos y Leyendas de Latinoamérica")
st.write("""
**Milela** integra, analiza y recomienda mitos y leyendas latinoamericanos
usando técnicas de **procesamiento del lenguaje natural (NLP)**.
Permite explorar, buscar y descubrir historias de toda la región.
""")

# 🔹 AGREGAMOS UNA NUEVA PESTAÑA PARA EL CHATBOT
tabs = st.tabs([
    "🔍 Buscar por temática",
    "📖 Explorar mitos por país",
    "📋 Encuesta de preferencias",
    "🤖 Consultar con IA"  # 🔹 NUEVA PESTAÑA
])

# --- TAB 4: Consultar con IA ---
with tabs[3]:
    st.subheader("🤖 Pregúntale a la IA sobre mitos latinoamericanos")
    
    # Selector de modelo
    modelo_groq = st.selectbox(
        "Selecciona el modelo:",
        ["mixtral-8x7b-32768", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        help="Mixtral es equilibrado, Llama 3.3 es más potente"
    )
    
    # Opción 1: Pregunta general
    st.markdown("### Opción 1: Pregunta general")
    pregunta_general = st.text_area(
        "Haz una pregunta sobre mitología latinoamericana:",
        placeholder="Ej: ¿Qué mitos hablan sobre espíritus del agua?",
        height=100
    )
    
    if st.button("🚀 Consultar", key="btn_general"):
        if pregunta_general:
            with st.spinner(f"Consultando {modelo_groq}..."):
                respuesta = consultar_groq(pregunta_general, "", modelo_groq)
            st.success("✅ Respuesta de la IA:")
            st.write(respuesta)
        else:
            st.warning("⚠️ Por favor escribe una pregunta")
    
    st.divider()
    
    # Opción 2: Pregunta sobre un mito específico
    st.markdown("### Opción 2: Pregunta sobre un mito específico")
    
    # Selector de mito
    mito_seleccionado = st.selectbox(
        "Selecciona un mito:",
        df_artefactos["titulo"].unique()
    )
    
    # Mostrar info del mito seleccionado
    mito_info = df_artefactos[df_artefactos["titulo"] == mito_seleccionado].iloc[0]
    
    with st.expander("📜 Ver información del mito"):
        st.write(f"**País:** {mito_info['pais']}")
        st.write(f"**Región:** {mito_info['region']}")
        st.write(f"**Temas:** {mito_info['temas_top3_str']}")
        st.write(f"**Texto:** {mito_info['texto'][:300]}...")
    
    pregunta_especifica = st.text_area(
        f"Haz una pregunta sobre '{mito_seleccionado}':",
        placeholder="Ej: ¿Qué simboliza este mito? ¿Cuál es su origen?",
        height=100
    )
    
    if st.button("🚀 Consultar sobre este mito", key="btn_especifico"):
        if pregunta_especifica:
            # Preparar contexto completo del mito
            contexto = f"""
Título: {mito_info['titulo']}
País: {mito_info['pais']}
Región: {mito_info['region']}
Temas: {mito_info['temas_top3_str']}

Texto completo:
{mito_info['texto']}
"""
            with st.spinner(f"Consultando {modelo_groq}..."):
                respuesta = consultar_groq(pregunta_especifica, contexto, modelo_groq)
            
            st.success("✅ Respuesta de la IA:")
            st.write(respuesta)
        else:
            st.warning("⚠️ Por favor escribe una pregunta")

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
        "Chile": ["Caleuche", "La Prodigiosa Pincoya", "El Trauco"],
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
            st.divider()
            st.subheader("✨ Recomendaciones similares a tu mito favorito:")
            match = df_artefactos[df_artefactos["titulo"].str.lower() == mito_favorito.lower()]
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
