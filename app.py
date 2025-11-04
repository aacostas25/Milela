import streamlit as st

# ======================
# Descripción del proyecto
# ======================

st.title("MILELA 🌎✨")
st.header("Mitos y Leyendas de Latinoamérica")
st.write("""
**Milela** es un proyecto que busca **integrar, analizar y recomendar mitos y leyendas latinoamericanos**
usando herramientas de **procesamiento del lenguaje natural (NLP)**.
Su objetivo es preservar y difundir el patrimonio cultural de la región mediante un sistema de recomendación
que permita descubrir nuevas historias según los gustos de cada usuario.
""")

st.markdown("---")
st.subheader("📋 Encuesta de preferencias")

# ======================
# Encuesta interactiva (sin form)
# ======================

nombre = st.text_input("Nombre")
edad = st.number_input("Edad", min_value=5, max_value=120, step=1)

pais = st.selectbox(
    "País de origen",
    ["Argentina", "Bolivia", "Chile", "Colombia",
     "Ecuador", "México", "Perú", "Uruguay"],
    key="pais"
)

# Diccionario de mitos por país
mitos_por_pais = {
    "Argentina": ["El Familiar", "La Luz Mala", "El Pombero"],
    "Bolivia": ["La Kantuta", "El Ekeko", "La Viuda del Monte"],
    "Chile": ["El Caleuche", "La Pincoya", "El Trauco"],
    "Colombia": ["La Llorona", "El Mohán", "La Patasola"],
    "Ecuador": ["El Duende", "La Dama Tapada", "La Tunda"],
    "México": ["La Nahuala", "El Chupacabras", "La Llorona"],
    "Perú": ["El Tunche", "La Jarjacha", "El Pishtaco"],
    "Uruguay": ["El Lobizón", "La Luz Mala", "El Pombero"]
}

# Este selectbox se actualiza dinámicamente
mito_favorito = st.selectbox(
    "Mito o leyenda favorita",
    mitos_por_pais[pais],
    key="mito_favorito"
)

# Botón separado, no dentro de un formulario
if st.button("Enviar"):
    if nombre:
        st.success(f"Gracias {nombre}, tus datos fueron registrados.")
        st.write(f"""
        **Resumen de tus respuestas:**
        - Edad: {edad}
        - País: {pais}
        - Mito favorito: {mito_favorito}
        """)
        st.info("Próximamente, Milela te recomendará nuevas leyendas basadas en tus gustos.")
    else:
        st.warning("Por favor, ingresa tu nombre antes de enviar.")

st.markdown("---")
st.caption("Proyecto desarrollado por **Andrea Acosta y Alexandra Moraga** – Pontificia Universidad Católica de Chile, 2025")
