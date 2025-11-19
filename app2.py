import streamlit as st
import pandas as pd
import json, os, glob, unicodedata
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Optional, Iterable
from groq import Groq

# ======================
# Configuración de Groq
# ======================
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False
    st.warning("⚠️ No se pudo conectar a Groq. Agrega tu API key en secrets.")

def modernizar_mito(mito_data: dict, modelo: str = "llama-3.3-70b-versatile") -> str:
    """
    Moderniza un mito usando Groq con instrucciones específicas.
    """
    if not GROQ_AVAILABLE:
        return "Error: Groq no está configurado. Agrega tu API key."
    
    titulo = mito_data.get("titulo", "")
    pais = mito_data.get("pais", "")
    region = mito_data.get("region", "Región no especificada")
    texto_original = mito_data.get("texto", "")
    
    instrucciones = f"""Eres un asistente experto en mitos y leyendas latinoamericanas.

TAREA:
Reescribir el mito de forma contemporánea, manteniendo fidelidad cultural y geográfica.

RESTRICCIONES CULTURALES Y GEOGRÁFICAS:
- No cambies el país ni la región de origen: {pais}, {region}.
- No inventes paisajes incoherentes con ese lugar (por ejemplo, no hables de desiertos en Chiloé).
- Si describes el entorno, usa solo elementos compatibles con el mito original o típicos de la zona
  (mar, ríos, lluvias, islas, bosques, cordillera, etc., según corresponda).
- Si no estás seguro de un detalle geográfico, es mejor omitirlo que inventarlo.

REGLAS NARRATIVAS:
- Mantén personajes principales, conflicto central y moraleja.
- Usa un lenguaje claro y actual, pensando en adolescentes.
- La extensión debe ser similar al original (no acortes demasiado ni extiendas excesivamente).
- Conserva la esencia del mito pero hazlo accesible para lectores contemporáneos.
- No uses lenguaje coloquial excesivo, mantén respeto por la tradición.

MITO ORIGINAL:
Título: {titulo}
País: {pais}
Región: {region}

Texto:
{texto_original}

Ahora escribe la versión modernizada del mito manteniendo todas las restricciones anteriores:"""
    
    try:
        completion = groq_client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": instrucciones}],
            temperature=0.7,
            max_tokens=2048,  # Aumentado para mitos más largos
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
# Funciones existentes (búsqueda, recomendación, etc.)
# ======================
def buscar_mitos_por_texto(query, top_k=5):
    qv = sbert.encode([query], convert_to_numpy=True).astype("float32")
    qv = normalize(qv, norm="l2", axis=1)
    D, I = index.search(qv, top_k)
    resultados = df_artefactos.iloc[I[0]].copy()
    resultados["score"] = D[0]
    return resultados[["id","pais", "region", "titulo", "temas_top3_str", "score", "texto"]]

def _jaccard(a, b):
    A, B = set([s.lower() for s in a]), set([s.lower() for s in b])
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def normalize_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("utf-8")
    return t.lower().strip()

def recommend_similar_to_item(item_id: str, top_k = 5) -> pd.DataFrame:
    fetch_k = 200
    w_sem = 0.7
    w_topic = 0.25

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

tabs = st.tabs([
    "🔍 Buscar por temática",
    "📖 Explorar mitos por país",
    "📋 Encuesta de preferencias",
    "🤖 Modernizar mito con IA"  # 🔹 NUEVA PESTAÑA
])

# --- TAB 4: Modernizar mito ---
with tabs[3]:
    st.subheader("🤖 Moderniza un mito usando IA")
    
    if not GROQ_AVAILABLE:
        st.error("❌ Para usar esta función necesitas configurar tu API key de Groq en Settings > Secrets")
        st.code('GROQ_API_KEY = "tu_clave_aqui"')
    else:
        st.info("✨ Selecciona un mito y modernízalo manteniendo su esencia cultural")
        
        # Selector de modelo
        col1, col2 = st.columns([3, 1])
        with col1:
            # Selector de mito por país
            pais_seleccionado = st.selectbox(
                "1️⃣ Selecciona el país:",
                sorted(df["pais"].unique())
            )
        
        with col2:
            modelo_groq = st.selectbox(
                "Modelo:",
                ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
                help="Llama 3.3 70B es el más potente"
            )
        
        # Filtrar mitos por país
        mitos_pais = df[df["pais"] == pais_seleccionado]["titulo"].unique()
        
        mito_seleccionado = st.selectbox(
            "2️⃣ Selecciona el mito a modernizar:",
            mitos_pais
        )
        
        # Obtener datos completos del mito
        mito_data = df[df["titulo"] == mito_seleccionado].iloc[0].to_dict()
        
        # Mostrar mito original
        with st.expander("📜 Ver mito original", expanded=True):
            st.write(f"**Título:** {mito_data['titulo']}")
            st.write(f"**País:** {mito_data['pais']}")
            st.write(f"**Región:** {mito_data['region'] if mito_data['region'] else 'No especificada'}")
            st.write(f"**ID:** {mito_data['id']}")
            st.divider()
            st.write("**Texto original:**")
            st.write(mito_data['texto'])
        
        # Botón para modernizar
        if st.button("🚀 Modernizar mito", type="primary", use_container_width=True):
            with st.spinner(f"✨ Modernizando '{mito_seleccionado}' con {modelo_groq}..."):
                version_moderna = modernizar_mito(mito_data, modelo_groq)
            
            st.success("✅ ¡Mito modernizado exitosamente!")
            
            # Mostrar versión modernizada
            st.subheader("📖 Versión Modernizada")
            st.write(version_moderna)
            
            # Botones de descarga y compartir
            col1, col2 = st.columns(2)
            
            with col1:
                # Crear JSON con ambas versiones
                resultado_completo = {
                    "titulo_original": mito_data["titulo"],
                    "pais": mito_data["pais"],
                    "region": mito_data["region"],
                    "id": mito_data["id"],
                    "texto_original": mito_data["texto"],
                    "texto_modernizado": version_moderna,
                    "modelo_usado": modelo_groq
                }
                
                json_descarga = json.dumps(resultado_completo, ensure_ascii=False, indent=4)
                
                st.download_button(
                    label="💾 Descargar JSON",
                    data=json_descarga,
                    file_name=f"mito_modernizado_{mito_data['id']}.json",
                    mime="application/json"
                )
            
            with col2:
                # Descargar como texto plano
                texto_descarga = f"""MITO MODERNIZADO
================

Título Original: {mito_data['titulo']}
País: {mito_data['pais']}
Región: {mito_data['region']}
ID: {mito_data['id']}

VERSIÓN ORIGINAL:
{mito_data['texto']}

---

VERSIÓN MODERNIZADA:
{version_moderna}

---
Generado con: {modelo_groq}
"""
                st.download_button(
                    label="📄 Descargar TXT",
                    data=texto_descarga,
                    file_name=f"mito_modernizado_{mito_data['id']}.txt",
                    mime="text/plain"
                )

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
        sorted(df["pais"].unique()),
        key="explorar_pais"
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
    if st.button("Buscar", key="buscar_btn"):
        if query.strip():
            resultados = buscar_mitos_por_texto(query.lower(), top_k=5)
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
