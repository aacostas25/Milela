# 🌎 MILELA – Mitos y Leyendas de Latinoamérica

**Autores:** Andrea Acosta y Alexandra Moraga  
**Institución:** Pontificia Universidad Católica de Chile (PUC Chile)  
**Año:** 2025  

---

## 🧩 Descripción del proyecto

**Milela** (*Mitos y Leyendas de Latinoamérica*) es un proyecto de investigación y desarrollo que busca **preservar, analizar y recomendar mitos y leyendas latinoamericanos** mediante el uso de **técnicas de procesamiento del lenguaje natural (NLP)**.

El objetivo principal es **integrar fuentes culturales dispersas** —como Wikipedia, Wikidata y libros en PDF— en un corpus estructurado, que permita desarrollar **sistemas de recomendación semánticos** y **aplicaciones educativas** para explorar la riqueza narrativa de la región.

---

## 🎯 Objetivos

- **Recolectar y limpiar** mitos y leyendas de América Latina desde fuentes abiertas.
- **Construir un corpus estructurado** con metadatos relevantes (país, tipo, fuente, entidades culturales).
- **Aplicar modelos NLP clásicos y modernos** (TF-IDF, Sentence Transformers, FAISS) para medir similitud semántica.
- **Desarrollar un sistema de recomendación** que sugiera mitos relacionados según los gustos del usuario.
- **Explorar la modernización narrativa** mediante prompting o fine-tuning en modelos generativos.

---

## 🧠 Técnicas y herramientas

| Etapa | Técnicas / Herramientas |
|-------|--------------------------|
| Extracción | Wikipedia API, Wikidata SPARQL, PyMuPDF |
| Preprocesamiento | spaCy, regex, pandas |
| Representación semántica | TF-IDF, Sentence Transformers |
| Recuperación y recomendación | FAISS, Scikit-learn |
| Visualización | Streamlit, Plotly, Matplotlib |

---

## 🧱 Estructura del proyecto

```
milela/
├── data/                     # Corpus consolidado (CSV, TXT)
├── notebooks/                # Experimentos y análisis en Jupyter
├── src/                      # Código fuente de modelos y utilidades
│   ├── nlp/                  # Preprocesamiento y embeddings
│   ├── recommender/          # Sistemas de recomendación
│   └── streamlit_app/        # Interfaz interactiva
├── images/                   # Ilustraciones o material gráfico
├── requirements.txt
└── README.md
```

---

## 🎨 Aplicación interactiva

La aplicación **Streamlit** de Milela permite al usuario explorar el proyecto y responder una **encuesta cultural** que servirá como base para el sistema de recomendación:

- Ingreso de **nombre, edad y país**.
- Selección de **mito o leyenda favorita**, dependiente del país elegido.
- Generación de un perfil inicial de usuario para recomendaciones futuras.

Ejecuta la app localmente:

```bash
streamlit run app.py
```

---

## 🌐 Países analizados

Argentina, Bolivia, Chile, Colombia, Ecuador, México, Perú y Uruguay.

Cada país cuenta con un conjunto de **mitos y leyendas** recopilados de fuentes digitales, incluyendo Wikipedia, Wikidata y libros digitalizados.

---

## 🧪 Etapas del proyecto

| Etapa | Descripción | Producto |
|-------|--------------|-----------|
| 1 | Definición del problema, objetivos y justificación | Documento de propuesta |
| 2 | Recolección y limpieza del corpus | CSV/TXT base |
| 3 | Implementación del baseline léxico | Recomendador TF-IDF |
| 4 | Implementación del modelo semántico | Sistema FAISS + embeddings |
| 5 | Evaluación y métricas | Informe técnico |
| 6 | Modernización narrativa | Relatos generados por modelo |
| 7 | Presentación final | Video y aplicación interactiva |

---

## 💡 Diferenciadores

- Enfoque **cultural y educativo** sobre un dominio no comercial.  
- Construcción de un **corpus original y abierto**.  
- Uso combinado de **técnicas clásicas y modernas de NLP**.  
- Propuesta de **modernización narrativa** mediante IA.  

---

## ⚙️ Instalación

Clona el repositorio e instala las dependencias:

```bash
git clone https://github.com/<tu_usuario>/milela.git
cd milela
pip install -r requirements.txt
```

Ejecuta la aplicación de encuesta:

```bash
streamlit run app.py
```

---

## 📚 Fuentes de datos

- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)
- [Wikidata SPARQL](https://query.wikidata.org/)
- Libros digitalizados y fuentes académicas en formato PDF.

---

## 📈 Próximos pasos

- Integrar la base de datos completa de mitos y leyendas.  
- Implementar el motor de recomendación basado en similitud semántica.  
- Publicar la aplicación completa en **Streamlit Cloud**.  
- Ampliar el corpus a nuevos países latinoamericanos.  

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia **MIT**.  
El contenido cultural recopilado pertenece a dominio público o fuentes con acceso libre para uso académico.

---

## ✨ Créditos

Proyecto desarrollado por  
**Alexandra Moraga** y **Andrea Acosta**
Pontificia Universidad Católica de Chile – 2025

