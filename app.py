import io
import platform
import hashlib

import streamlit as st
from PIL import Image
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="DocuMind · Análisis PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Estilo ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #697386;
        --purple: #7057E8;
        --purple-dark: #5841CB;
        --line: #E8EAF2;
        --surface: #FFFFFF;
        --canvas: #F6F7FB;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(246,247,251,.88); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }

    [data-testid="stSidebar"] {
        background: #17172A;
        border-right: 1px solid rgba(255,255,255,.06);
    }
    [data-testid="stSidebar"] * { color: #F4F3FF; }
    [data-testid="stSidebar"] .stCaption { color: #AAA9C5 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.13); }

    .brand {
        display: flex; align-items: center; gap: 12px;
        margin: .35rem 0 2rem 0;
    }
    .brand-icon {
        width: 44px; height: 44px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg,#8E7BFF,#5B43D6);
        color: white; font-size: 22px; box-shadow: 0 8px 24px rgba(112,87,232,.3);
    }
    .brand-name { font: 800 19px 'Manrope',sans-serif; letter-spacing: -.5px; }
    .brand-sub { color: #AAA9C5; font-size: 11px; margin-top: 1px; }
    .eyebrow {
        color: var(--purple); font-size: 11px; font-weight: 700;
        letter-spacing: 1.8px; text-transform: uppercase; margin-bottom: 8px;
    }
    .hero-title {
        font: 800 clamp(28px,3.2vw,40px) 'Manrope',sans-serif;
        letter-spacing: -1.5px; line-height: 1.15; color: #171B2D;
        margin: 0 0 10px 0;
    }
    .hero-copy { color: var(--muted); font-size: 15px; line-height: 1.65; max-width: 720px; }
    .top-pill {
        display: inline-flex; align-items: center; gap: 7px; border-radius: 999px;
        padding: 7px 11px; border: 1px solid #E3DFFD; background: #F0EDFF;
        color: #5A43C7; font-size: 12px; font-weight: 700;
    }
    .section-heading {
        font: 700 17px 'Manrope',sans-serif; letter-spacing: -.3px;
        color: #20263A; margin: 0 0 4px 0;
    }
    .section-copy { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
    .metric-card {
        background: white; border: 1px solid var(--line); border-radius: 18px;
        padding: 19px 20px; min-height: 116px;
        box-shadow: 0 3px 12px rgba(25,32,60,.025);
    }
    .metric-label { color: #788094; font-size: 12px; font-weight: 600; margin-bottom: 9px; }
    .metric-value { color: #1B2033; font: 800 26px 'Manrope',sans-serif; letter-spacing: -.8px; }
    .metric-note { color: #9299A9; font-size: 11px; margin-top: 4px; }
    .panel {
        background: white; border: 1px solid var(--line); border-radius: 20px;
        padding: 22px 24px; box-shadow: 0 4px 18px rgba(25,32,60,.03);
        margin-bottom: 18px;
    }
    .panel-title { font: 700 16px 'Manrope',sans-serif; color: #20263A; margin-bottom: 5px; }
    .panel-subtitle { font-size: 13px; color: var(--muted); margin-bottom: 17px; }
    .empty-state {
        border: 1px dashed #D8D9E8; background: rgba(255,255,255,.72);
        border-radius: 20px; padding: 38px 24px; text-align: center;
    }
    .empty-icon {
        display: inline-flex; width: 58px; height: 58px; border-radius: 18px;
        align-items: center; justify-content: center; font-size: 27px;
        background: #F0EDFF; margin-bottom: 13px;
    }
    .empty-title { font: 700 17px 'Manrope',sans-serif; color: #242A3D; margin-bottom: 6px; }
    .empty-copy { color: #7A8295; font-size: 13px; line-height: 1.6; }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #C9C2F5; border-radius: 15px; padding: 12px;
        background: #FBFAFF;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        border-radius: 12px !important; border-color: #E1E4EF !important;
        background: #FCFCFE !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #8D7AF0 !important;
        box-shadow: 0 0 0 2px rgba(112,87,232,.12) !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg,#7962EE,#6047D5);
        color: white; border: 0; border-radius: 12px; padding: .65rem 1.1rem;
        font-weight: 700; min-height: 44px;
        box-shadow: 0 5px 14px rgba(100,75,216,.18);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        color: white; border: 0; transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(100,75,216,.25);
    }
    div[data-testid="stExpander"] {
        background: white; border: 1px solid var(--line); border-radius: 13px;
    }
    .answer-label {
        display: inline-flex; align-items: center; gap: 7px;
        background: #EDEAFF; color: #5943C7; border-radius: 8px;
        padding: 6px 9px; font-size: 11px; font-weight: 800;
        letter-spacing: .6px; margin-bottom: 12px;
    }
    .footer-note { text-align: center; color: #9AA0AF; font-size: 11px; padding-top: 14px; }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .panel { padding: 17px; }
        .hero-title { letter-spacing: -1px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Estado ----------
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_pdf_hash" not in st.session_state:
    st.session_state.last_pdf_hash = None

# ---------- Barra lateral ----------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div>
                <div class="brand-name">DocuMind</div>
                <div class="brand-sub">INTELIGENCIA DOCUMENTAL</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Tu espacio de trabajo")
    st.caption("Carga un documento y conversa con su contenido usando IA.")
    st.divider()
    st.markdown("**① · Conecta tu API**")
    api_key = st.text_input(
        "Clave de OpenAI",
        type="password",
        placeholder="sk-••••••••••••",
        help="La clave se utiliza para generar embeddings y respuestas.",
        label_visibility="collapsed",
    )
    st.caption("Tu clave no se muestra en pantalla.")
    st.divider()
    st.markdown("**② · Añade un documento**")
    pdf = st.file_uploader(
        "Selecciona un PDF",
        type=["pdf"],
        help="El documento debe contener texto seleccionable. Los PDF escaneados pueden requerir OCR.",
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Sobre DocuMind**")
    st.caption("Consulta documentos extensos y encuentra respuestas basadas en su contenido.")
    st.caption(f"Python {platform.python_version()}")

# ---------- Encabezado ----------
head_left, head_right = st.columns([5, 1.5], vertical_alignment="center")
with head_left:
    st.markdown('<div class="eyebrow">TU ASISTENTE DOCUMENTAL</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Habla con tus documentos.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Convierte archivos PDF en conocimiento consultable. '
        'Haz preguntas concretas y obtén respuestas basadas en el contenido de tu documento.</div>',
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown('<div class="top-pill">✦ IA ACTIVADA</div>', unsafe_allow_html=True)

st.write("")

# ---------- Índice ----------
@st.cache_resource(show_spinner=False)
def crear_base_conocimiento(pdf_bytes: bytes, api_key: str):
    """Extrae texto y crea el índice vectorial, reutilizado por caché."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texto = "\n".join((pagina.extract_text() or "") for pagina in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_text(texto)
    if not fragmentos:
        return None, len(texto), 0, len(reader.pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    base = FAISS.from_texts(fragmentos, embeddings)
    return base, len(texto), len(fragmentos), len(reader.pages)

# ---------- Estado inicial ----------
if not api_key or pdf is None:
    st.markdown('<div class="section-heading">Resumen del espacio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Tu actividad documental aparecerá aquí.</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, label, value, note in [
        (m1, "DOCUMENTO", "Pendiente", "Carga un archivo PDF"),
        (m2, "CONOCIMIENTO", "—", "Se indexará automáticamente"),
        (m3, "CONSULTAS", "Listo para empezar", "Tu pregunta será el siguiente paso"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="font-size:21px">{value}</div>'
                f'<div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    if not api_key:
        title = "Conecta tu clave de OpenAI"
        copy = "Añade tu clave en la barra lateral para preparar el análisis documental."
        icon = "🔑"
    else:
        title = "Tu próximo documento empieza aquí"
        copy = "Carga un archivo PDF desde la barra lateral. Lo convertiremos en fragmentos consultables."
        icon = "📄"
    st.markdown(
        f'<div class="empty-state"><div class="empty-icon">{icon}</div>'
        f'<div class="empty-title">{title}</div><div class="empty-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )
else:
    pdf_bytes = pdf.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Evita mostrar una respuesta vieja cuando cambia el documento.
    if st.session_state.last_pdf_hash != pdf_hash:
        st.session_state.last_answer = None
        st.session_state.last_question = None
        st.session_state.last_pdf_hash = pdf_hash

    try:
        with st.spinner("Preparando el conocimiento del documento…"):
            base, n_caracteres, n_fragmentos, n_paginas = crear_base_conocimiento(pdf_bytes, api_key)

        if base is None:
            st.error("No se encontró texto extraíble. Si es un PDF escaneado, necesitas aplicar OCR primero.")
            st.stop()

        st.markdown('<div class="section-heading">Resumen del documento</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">Tu archivo ya está listo para consultas.</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        metrics = [
            (m1, "DOCUMENTO", pdf.name, f"{n_paginas} páginas"),
            (m2, "TEXTO EXTRAÍDO", f"{n_caracteres:,}", "Caracteres indexados"),
            (m3, "BASE DE CONOCIMIENTO", f"{n_fragmentos:,}", "Fragmentos disponibles"),
        ]
        for col, label, value, note in metrics:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value" style="font-size:{"19px" if len(str(value)) > 18 else "26px"}">{value}</div>'
                    f'<div class="metric-note">{note}</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            st.markdown('<div class="panel-title">✦ Haz una pregunta</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-subtitle">Pregunta por conceptos, cifras, conclusiones o detalles del documento.</div>',
                unsafe_allow_html=True,
            )
            with st.form("consulta_form", clear_on_submit=False):
                pregunta = st.text_area(
                    "Tu pregunta",
                    placeholder="Ej. ¿Cuáles son las conclusiones principales del documento?",
                    height=125,
                    label_visibility="collapsed",
                )
                enviar = st.form_submit_button("Consultar documento  →", use_container_width=True)

            if enviar:
                if not pregunta.strip():
                    st.warning("Escribe una pregunta antes de consultar.")
                else:
                    try:
                        with st.spinner("Buscando información y redactando la respuesta…"):
                            docs = base.similarity_search(pregunta.strip(), k=4)
                            contexto = "\n\n---\n\n".join(d.page_content for d in docs)
                            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
                            mensajes = [
                                (
                                    "system",
                                    "Responde en español usando únicamente el contexto proporcionado. "
                                    "Si la respuesta no está en el contexto, dilo claramente. "
                                    "No inventes datos. Distingue hechos de inferencias."
                                ),
                                ("human", f"Contexto:\n{contexto}\n\nPregunta: {pregunta.strip()}"),
                            ]
                            respuesta = llm.invoke(mensajes)
                        st.session_state.last_answer = {
                            "question": pregunta.strip(),
                            "content": respuesta.content,
                            "docs": [d.page_content for d in docs],
                        }
                        st.session_state.last_question = pregunta.strip()
                    except Exception:
                        st.error("No se pudo completar la consulta. Verifica tu clave, conexión y límites de uso de OpenAI.")

        with right:
            st.markdown('<div class="panel-title">Cómo obtener mejores respuestas</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div style="padding:14px 0;border-bottom:1px solid #ECEEF5">
                    <div style="font-weight:700;color:#292E42;font-size:13px">🎯 Sé específico</div>
                    <div style="color:#7A8295;font-size:12px;margin-top:5px;line-height:1.55">Pregunta por un tema, una sección o una conclusión concreta.</div>
                </div>
                <div style="padding:14px 0;border-bottom:1px solid #ECEEF5">
                    <div style="font-weight:700;color:#292E42;font-size:13px">📌 Pide evidencia</div>
                    <div style="color:#7A8295;font-size:12px;margin-top:5px;line-height:1.55">Solicita que la respuesta distinga los datos explícitos de las interpretaciones.</div>
                </div>
                <div style="padding:14px 0">
                    <div style="font-weight:700;color:#292E42;font-size:13px">🔎 Revisa el contexto</div>
                    <div style="color:#7A8295;font-size:12px;margin-top:5px;line-height:1.55">Consulta los fragmentos recuperados para comprobar de dónde sale la respuesta.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # La respuesta persiste aunque Streamlit vuelva a ejecutar el script.
        if st.session_state.last_answer:
            answer = st.session_state.last_answer
            st.write("")
            st.markdown('<div class="panel-title">Respuesta generada</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-label">✦ RESPUESTA DE IA</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Tu pregunta:** {answer['question']}")
            st.markdown(answer["content"])
            with st.expander("Ver fragmentos utilizados como contexto"):
                for i, fragmento in enumerate(answer["docs"], 1):
                    st.markdown(f"**Fragmento {i}**")
                    st.text(fragmento)
                    if i < len(answer["docs"]):
                        st.divider()

    except Exception:
        st.error("No se pudo procesar el PDF. Comprueba que el archivo sea válido y que tu clave de OpenAI funcione.")
        with st.expander("Detalles técnicos"):
            st.caption("Revisa la consola del servidor para ver el error completo. Evita compartir claves de API en capturas o registros.")

st.markdown('<div class="footer-note">DOCUMIND · Análisis documental con IA · Las respuestas deben verificarse contra el documento original.</div>', unsafe_allow_html=True)
