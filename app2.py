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
    Se basa EXCLUSIVAMENTE en el texto proporcionado.
    """
    if not GROQ_AVAILABLE:
        return "Error: Groq no está configurado. Agrega tu API key."
    
    titulo = mito_data.get("titulo", "")
    pais = mito_data.get("pais", "")
    region = mito_data.get("region", "Región no especificada")
    texto_original = mito_data.get("texto", "")
    
    instrucciones = f"""Eres un asistente experto en mitos y leyendas latinoamericanas.

IMPORTANTE - RESTRICCIÓN FUNDAMENTAL:
Debes basarte EXCLUSIVAMENTE en el texto del mito que te proporciono a continuación.
NO agregues información externa, NO uses tu conocimiento previo sobre este mito.
Si el texto original no menciona algo, NO lo inventes ni lo agregues.

TAREA:
Reescribir el mito de forma contemporánea, manteniendo fidelidad cultural y geográfica.

RESTRICCIONES CULTURALES Y GEOGRÁFICAS:
- No cambies el país ni la región de origen: {pais}, {region}.
- No inventes paisajes incoherentes con ese lugar (por ejemplo, no hables de desiertos en Chiloé).
- Si describes el entorno, usa SOLO elementos mencionados en el texto original o elementos típicos obvios de la zona.
- Si no estás seguro de un detalle geográfico, es mejor omitirlo que inventarlo.
- NO agregues datos históricos o culturales que no estén en el texto original.

REGLAS NARRATIVAS:
- Mantén SOLO los personajes que aparecen en el texto original proporcionado.
- Mantén SOLO el conflicto central descrito en el texto original.
- Mantén SOLO la moraleja o conclusión presente en el texto original.
- Usa un lenguaje claro y actual, pensando en adolescentes.
- La extensión debe ser similar al original (no acortes demasiado ni extiendas excesivamente).
- Conserva la esencia del mito pero hazlo accesible para lectores contemporáneos.
- No uses lenguaje coloquial excesivo, mantén respeto por la tradición.
- NO inventes diálogos, eventos o detalles que no estén en el texto original.

TEXTO ORIGINAL DEL MITO (tu ÚNICA fuente de información):
---
Título: {titulo}
País: {pais}
Región: {region}

{texto_original}
---

Ahora reescribe SOLO lo que está en el texto anterior, modernizando el lenguaje pero sin agregar información nueva.
No uses frases como "según la leyenda" o "se dice que" - escribe directamente la historia modernizada.
Escribe la versión modernizada:"""
    
    try:
        completion = groq_client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": instrucciones}],
            temperature=0.5,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al consultar Groq: {str(e)}"

# 🔹 NUEVA FUNCIÓN: Crear mito con prompt personalizado
def crear_mito_personalizado(mito_data: dict, prompt_usuario: str, modelo: str = "llama-3.3-70b-versatile") -> str:
    """
    Crea versión personalizada del mito basándose en instrucciones del usuario.
    Se basa EXCLUSIVAMENTE en el texto proporcionado.
    """
    if not GROQ_AVAILABLE:
        return "Error: Groq no está configurado. Agrega tu API key."
    
    titulo = mito_data.get("titulo", "")
    pais = mito_data.get("pais", "")
    region = mito_data.get("region", "Región no especificada")
    texto_original = mito_data.get("texto", "")
    
    instrucciones = f"""Eres un asistente experto en mitos y leyendas latinoamericanas.

RESTRICCIÓN FUNDAMENTAL:
Debes basarte EXCLUSIVAMENTE en el texto del mito que te proporciono.
NO agregues información que no esté en el texto original.
NO uses tu conocimiento previo sobre este mito.

INFORMACIÓN DEL MITO ORIGINAL:
---
Título: {titulo}
País: {pais}
Región: {region}

Texto original:
{texto_original}
---

INSTRUCCIONES DEL USUARIO:
{prompt_usuario}

REGLAS IMPORTANTES:
1. Respeta el país y región de origen: {pais}, {region}
2. No inventes elementos geográficos incoherentes con la ubicación
3. Usa SOLO información presente en el texto original
4. Si el usuario pide algo que contradice el texto original, prioriza la fidelidad al mito
5. Si el usuario pide agregar elementos no mencionados, explica que no están en el original

Ahora crea la versión del mito siguiendo las instrucciones del usuario:"""
    
    try:
        completion = groq_client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": instrucciones}],
            temperature=0.6,  # Un poco más de creatividad que la versión estándar
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al consultar Groq: {str(e)}"

# ======================
# Carga de datos (sin cambios)
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
    "✨ Modernizar mito"  # 🔹 NUEVA PESTAÑA
])

# 🔹 --- TAB 5: Crea tu mito ---
with tabs[3]:
    st.subheader("✨ Crea tu versión personalizada de un mito")
    
    if not GROQ_AVAILABLE:
        st.error("❌ Para usar esta función necesitas configurar tu API key de Groq en Settings > Secrets")
        st.code('GROQ_API_KEY = "tu_clave_aqui"')
    else:
        st.info("🎨 Selecciona un mito y escribe tus propias instrucciones para crear una versión personalizada")
        
        # Selector de modelo
        col1, col2 = st.columns([3, 1])
        with col1:
            pais_seleccionado_crear = st.selectbox(
                "1️⃣ Selecciona el país:",
                sorted(df["pais"].unique()),
                key="crear_pais"
            )
        
        with col2:
            modelo_groq_crear = st.selectbox(
                "Modelo:",
                ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
                key="crear_modelo",
                help="Llama 3.3 70B es el más potente"
            )
        
        # Filtrar mitos por país
        mitos_pais_crear = df[df["pais"] == pais_seleccionado_crear]["titulo"].unique()
        
        mito_seleccionado_crear = st.selectbox(
            "2️⃣ Selecciona el mito base:",
            mitos_pais_crear,
            key="crear_mito"
        )
        
        # Obtener datos completos del mito
        mito_data_crear = df[df["titulo"] == mito_seleccionado_crear].iloc[0].to_dict()
        
        # Mostrar mito original
        with st.expander("📜 Ver mito original"):
            st.write(f"**Título:** {mito_data_crear['titulo']}")
            st.write(f"**País:** {mito_data_crear['pais']}")
            st.write(f"**Región:** {mito_data_crear['region'] if mito_data_crear['region'] else 'No especificada'}")
            st.divider()
            st.write(mito_data_crear['texto'])
        
        st.markdown("### 3️⃣ Escribe tus instrucciones personalizadas")
        
        # Ejemplos predefinidos
        with st.expander("💡 Ver ejemplos de instrucciones"):
            st.markdown("""
            **Ejemplos para profesores:**
            - "Reescribe el mito como si fuera una noticia de periódico actual, manteniendo los eventos principales"
            - "Adapta el mito para niños de 8-10 años, usando un lenguaje muy simple y añadiendo descripciones visuales"
            - "Cuenta el mito desde el punto de vista de uno de los personajes secundarios"
            
            **Ejemplos creativos:**
            - "Reescribe el mito en forma de diálogo entre los personajes principales"
            - "Adapta el mito al contexto urbano moderno, pero manteniendo la moraleja original"
            
            **Ejemplos educativos:**
            - "Reescribe el mito destacando los valores morales y explicándolos claramente"
            - "Adapta el mito para enseñar sobre la importancia del respeto a la naturaleza"
            - "Escribe el mito en formato de preguntas y respuestas para comprensión lectora"
            """)
        
        # Campo de texto para instrucciones personalizadas
        prompt_personalizado = st.text_area(
            "Escribe tus instrucciones aquí:",
            height=150,
            placeholder="Ejemplo: Reescribe el mito como una carta que el personaje principal le escribe a su familia contando lo que vivió...",
            help="Sé específico sobre qué quieres: estilo, formato, público objetivo, elementos a destacar, etc."
        )
        
        # Opciones adicionales
        col1, col2 = st.columns(2)
        with col1:
            incluir_original = st.checkbox("Mostrar texto original junto a la versión personalizada", value=True)
        with col2:
            longitud_preferida = st.select_slider(
                "Longitud aproximada:",
                options=["Muy breve", "Breve", "Similar al original", "Extendida", "Muy detallada"],
                value="Similar al original"
            )
        
        # Agregar longitud al prompt si el usuario la seleccionó
        if longitud_preferida != "Similar al original":
            prompt_con_longitud = f"{prompt_personalizado}\n\nLongitud deseada: {longitud_preferida}"
        else:
            prompt_con_longitud = prompt_personalizado
        
        # Botón para generar
        if st.button("🚀 Generar versión personalizada", type="primary", use_container_width=True, key="btn_crear"):
            if not prompt_personalizado.strip():
                st.warning("⚠️ Por favor escribe tus instrucciones antes de generar")
            else:
                with st.spinner(f"✨ Creando versión personalizada de '{mito_seleccionado_crear}'..."):
                    version_personalizada = crear_mito_personalizado(
                        mito_data_crear, 
                        prompt_con_longitud, 
                        modelo_groq_crear
                    )
                
                st.success("✅ ¡Versión personalizada creada exitosamente!")
                
                # Mostrar resultados
                if incluir_original:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📜 Versión Original")
                        st.write(mito_data_crear['texto'])
                    with col2:
                        st.subheader("✨ Versión Personalizada")
                        st.write(version_personalizada)
                else:
                    st.subheader("✨ Versión Personalizada")
                    st.write(version_personalizada)
                
                st.divider()
                
                # Botones de descarga
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Descargar JSON completo
                    resultado_completo = {
                        "titulo_original": mito_data_crear["titulo"],
                        "pais": mito_data_crear["pais"],
                        "region": mito_data_crear["region"],
                        "id": mito_data_crear["id"],
                        "texto_original": mito_data_crear["texto"],
                        "instrucciones_usuario": prompt_personalizado,
                        "texto_personalizado": version_personalizada,
                        "modelo_usado": modelo_groq_crear,
                        "longitud_preferida": longitud_preferida
                    }
                    
                    json_descarga = json.dumps(resultado_completo, ensure_ascii=False, indent=4)
                    
                    st.download_button(
                        label="💾 Descargar JSON",
                        data=json_descarga,
                        file_name=f"mito_personalizado_{mito_data_crear['id']}.json",
                        mime="application/json",
                        key="download_json_crear"
                    )
                
                with col2:
                    # Descargar solo versión personalizada (TXT)
                    st.download_button(
                        label="📄 Descargar versión personalizada (TXT)",
                        data=version_personalizada,
                        file_name=f"mito_personalizado_{mito_data_crear['id']}.txt",
                        mime="text/plain",
                        key="download_txt_crear"
                    )
                
                with col3:
                    # Descargar comparación completa
                    texto_comparacion = f"""VERSIÓN PERSONALIZADA DEL MITO
================================

Título Original: {mito_data_crear['titulo']}
País: {mito_data_crear['pais']}
Región: {mito_data_crear['region']}
ID: {mito_data_crear['id']}

INSTRUCCIONES DEL USUARIO:
{prompt_personalizado}

---

VERSIÓN ORIGINAL:
{mito_data_crear['texto']}

---

VERSIÓN PERSONALIZADA:
{version_personalizada}

---
Generado con: {modelo_groq_crear}
Longitud preferida: {longitud_preferida}
"""
                    st.download_button(
                        label="📋 Descargar comparación completa",
                        data=texto_comparacion,
                        file_name=f"comparacion_{mito_data_crear['id']}.txt",
                        mime="text/plain",
                        key="download_comparacion"
                    )

# --- TAB 3: Encuesta (sin cambios) ---
with tabs[2]:
    st.subheader("Encuesta de preferencias")
    pais = st.selectbox("País de origen", 
                        ["Argentina","Bolivia","Chile","Colombia","Ecuador","México","Perú","Uruguay"],
                        key="encuesta_pais")

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

    if st.button("Enviar", key="btn_encuesta"):
        if nombre:
            st.success(f"Gracias {nombre}, tus datos fueron registrados.")
            st.write(f"""
            **Resumen de tus respuestas:**
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

# --- TAB 2: Exploración (sin cambios) ---
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

# --- TAB 1: Buscador (sin cambios) ---
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
