import streamlit as st
import cv2
from fer import FER
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import random
import db
import tempfile
import os
import json

# Page configuration
st.set_page_config(
    page_title="Detector emocional",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .emotion-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    h1 {
        color: #FF6B6B;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .report-section {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session state initialization
# ---------------------------
if 'page' not in st.session_state:
    st.session_state.page = "Control"
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = pd.DataFrame()
if 'detector' not in st.session_state:
    st.session_state.detector = FER(mtcnn=True)
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'visible_emotions' not in st.session_state:
    st.session_state.visible_emotions = {
        'happy': True, 'sad': True, 'angry': True,
        'surprise': True, 'fear': True, 'disgust': True, 'neutral': True
    }
if 'last_detected_emotions' not in st.session_state:
    st.session_state.last_detected_emotions = None
if 'show_report' not in st.session_state:
    st.session_state.show_report = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Emotion configuration
EMOTION_COLORS = {
    'happy': '#FFE48C',
    'sad': '#91CEFF',
    'angry': '#FF5656',
    'surprise': '#4DFFBE',
    'fear': '#DF7BDF',
    'disgust': '#FFA500',
    'neutral': '#4B4B4B'
}

EMOTION_NAMES = {
    'happy': 'Feliz',
    'sad': 'Triste',
    'angry': 'Enojo',
    'surprise': 'Sorpresa',
    'fear': 'Miedo',
    'disgust': 'Disgusto',
    'neutral': 'Neutral'
}

EMOTIONS_ORDER = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

# ---------------------------
# Helper utilities
# ---------------------------
def create_emotion_dataframe(emotions_dict, timestamp):
    row = {'timestamp': timestamp, 'time_seconds': 0}
    for emotion in EMOTIONS_ORDER:
        row[emotion] = emotions_dict.get(emotion, 0)
    return pd.DataFrame([row])

def detect_emotions_from_frame(frame):
    try:
        results = st.session_state.detector.detect_emotions(frame)
        if results and len(results) > 0:
            return results[0]['emotions'], results[0]['box']
        return None, None
    except Exception as e:
        st.error(f"Error detecting emotions: {str(e)}")
        return None, None

def create_realtime_graph(df):
    if df.empty or len(df) < 2:
        return None
    fig = go.Figure()
    for emotion in EMOTIONS_ORDER:
        if emotion in df.columns and st.session_state.visible_emotions.get(emotion, True):
            fig.add_trace(go.Scatter(
                x=df['time_seconds'],
                y=df[emotion],
                mode='lines',
                name=EMOTION_NAMES[emotion],
                line=dict(color=EMOTION_COLORS[emotion], width=2),
                hovertemplate=f'<b>{EMOTION_NAMES[emotion]}</b><br>Time: %{{x:.1f}}s<br>Intensity: %{{y:.2f}}<extra></extra>',
                visible=True
            ))
    fig.update_layout(
        title="Evolución de Emociones",
        xaxis_title="Tiempo (segundos)",
        yaxis_title="Intensidad",
        hovermode='x unified',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#0E1117',
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    fig.update_xaxes(gridcolor='#333333')
    fig.update_yaxes(gridcolor='#333333')
    return fig

def create_emotion_bars(emotions):
    fig = go.Figure()
    emotions_sorted = [(EMOTION_NAMES[e], emotions.get(e, 0), EMOTION_COLORS[e]) for e in EMOTIONS_ORDER]
    for name, value, color in emotions_sorted:
        fig.add_trace(go.Bar(
            y=[name],
            x=[value],
            orientation='h',
            marker=dict(color=color),
            text=f'{value*100:.1f}%',
            textposition='auto',
            hovertemplate=f'<b>{name}</b>: %{{x:.2f}}<extra></extra>'
        ))
    fig.update_layout(
        title="Desglose Actual de Emociones",
        xaxis_title="Intensidad",
        showlegend=False,
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#0E1117',
        font=dict(color='white'),
        height=400,
        xaxis=dict(range=[0, 1])
    )
    return fig

def create_pie_chart(df):
    if df.empty:
        return None
    emotion_avgs = {}
    for emotion in EMOTIONS_ORDER:
        if emotion in df.columns:
            emotion_avgs[EMOTION_NAMES[emotion]] = df[emotion].mean()
    emotion_avgs = {k: v for k, v in emotion_avgs.items() if v > 0.01}
    if not emotion_avgs:
        return None
    # map names back to keys for colors
    keys_for_colors = []
    for name in emotion_avgs.keys():
        for k, v in EMOTION_NAMES.items():
            if v == name:
                keys_for_colors.append(k)
                break
    fig = go.Figure(data=[go.Pie(
        labels=list(emotion_avgs.keys()),
        values=list(emotion_avgs.values()),
        marker=dict(colors=[EMOTION_COLORS[k] for k in keys_for_colors]),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>'
    )])
    fig.update_layout(
        title="Distribución de Emociones",
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#0E1117',
        font=dict(color='white'),
        height=400
    )
    return fig

def generate_report_data(df, session_id, start_time, end_time):
    if df.empty:
        return None
    duration = end_time - start_time
    emotion_stats = {}
    for emotion in EMOTIONS_ORDER:
        if emotion in df.columns:
            emotion_stats[EMOTION_NAMES[emotion]] = {
                'promedio': df[emotion].mean() * 100,
                'maximo': df[emotion].max() * 100,
                'detecciones_altas': int((df[emotion] > 0.5).sum())
            }
    return {
        'session_id': session_id,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'total_frames': len(df),
        'emotion_stats': emotion_stats
    }

def display_session_report(report_data, emotion_df):
    """
    Toma un diccionario de reporte y un dataframe de emociones
    y los muestra en la UI de Streamlit.
    """
    
    # --- 1. Preparar datos (Maneja datos vivos o de JSON) ---
    
    # Duración (puede ser 'timedelta' o un número en segundos)
    duration_val = report_data.get('duration', 0)
    if isinstance(duration_val, (int, float)):
        duration_obj = timedelta(seconds=duration_val)
    elif isinstance(duration_val, timedelta):
        duration_obj = duration_val
    else:
        duration_obj = timedelta(seconds=0) # Fallback
    duration_str = str(duration_obj).split('.')[0] # Formato HH:MM:SS

    # Tiempo de inicio (puede ser 'datetime' o un string ISO)
    start_time_val = report_data.get('start_time')
    if isinstance(start_time_val, str):
        try:
            start_time_obj = datetime.fromisoformat(start_time_val)
        except ValueError:
            start_time_obj = datetime.now() # Fallback
    elif isinstance(start_time_val, datetime):
        start_time_obj = start_time_val
    else:
        start_time_obj = datetime.now() # Fallback
    start_time_str = start_time_obj.strftime('%H:%M:%S')

    # --- 2. Mostrar la UI ---
    
    st.markdown(f"### Reporte de Sesión")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ID de Sesión", report_data.get('session_id', 'N/A'))
    with col2:
        st.metric("Duración", duration_str)
    with col3:
        st.metric("Frames Totales", report_data.get('total_frames', 0))
    with col4:
        st.metric("Inicio", start_time_str)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        line_graph = create_realtime_graph(emotion_df)
        if line_graph:
            st.plotly_chart(line_graph, use_column_width=True)
    with col2:
        pie_chart = create_pie_chart(emotion_df)
        if pie_chart:
            st.plotly_chart(pie_chart, use_column_width=True)

    st.markdown("---")
    st.subheader("Análisis por Emoción")
    
    emotion_stats = report_data.get('emotion_stats', {})
    if not emotion_stats:
        st.warning("No hay estadísticas de emociones en este reporte.")
        return

    for emotion_name, stats in emotion_stats.items():
        st.write(f"**{emotion_name}**")
        avg_val = stats.get('promedio', 0) / 100
        st.progress(avg_val)
        st.text(f"Promedio: {stats.get('promedio', 0):.2f}%   |   Máximo: {stats.get('maximo', 0):.2f}%   |   Detecciones altas (>50%): {stats.get('detecciones_altas', 0)}")
        st.markdown("---")

# ---------------------------
# Sidebar: page selection and patient management
# ---------------------------
with st.sidebar:
    st.title("Panel de control")
    st.markdown("---")
    st.session_state.page = st.selectbox("Página", ["Control", "Historial", "Administrar"], index=["Control", "Historial", "Administrar"].index(st.session_state.page) if st.session_state.page in ["Control", "Historial", "Administrar"] else 0)
    st.markdown("---")
    def registrar_usuario():
        # Leemos el nombre desde el session_state
        patient_name_to_add = st.session_state.new_patient_name_input.strip()
        
        if patient_name_to_add:
            try:
                user_id = db.create_user(patient_name_to_add)
                st.success("Usuario registrado")
                st.session_state.user_id = user_id
                st.session_state.user_name = patient_name_to_add
                
                # 2. ¡El truco! Limpiamos el valor AQUÍ
                st.session_state.new_patient_name_input = "" 
                
            except Exception as e:
                st.error(f"Error al registrar usuario: {e}")
        else:
            st.warning("Escribe un nombre válido")
    # Load patients
    try:
        patients = db.get_users()
    except Exception as e:
        patients = []
        st.error("Error cargando Usuarios desde la base de datos. Revisa la conexión.")
    patient_names = [p["name"] for p in patients]

    st.markdown("## Usuarios")
    selected_patient = st.selectbox("Selecciona usuario", ["-- Nuevo / Ninguno --"] + patient_names)
    
    if 'new_patient_name_input' not in st.session_state:
        st.session_state.new_patient_name_input = ""

    new_patient_name = st.text_input("Registrar nuevo usuario", key='new_patient_name_input')

    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "Registrar usuario",
            on_click=registrar_usuario
        )
    with col2:
        if st.button("Recargar Usuarios"):
            st.rerun()

    # Set user_id if selected from list
    if selected_patient in patient_names:
        user = next((p for p in patients if p["name"] == selected_patient), None)
        if user:
            st.session_state.user_id = user["id"]
            st.session_state.user_name = user["name"]

    st.markdown("---")
    st.markdown("Modo")
    mode = st.radio("Seleccionar modo:", ["Webcam en vivo", "Subir archivo"])

# ---------------------------
# Main layout title
# ---------------------------
st.title("Detector de Emociones en Tiempo Real")
st.markdown("---")

# ---------------------------
# Page: CONTROL (Live capture + saving)
# ---------------------------
if st.session_state.page == "Control":
    if mode == "Webcam en vivo":
        st.subheader("Control en Vivo")

        # Start / Stop
        if not st.session_state.session_active:
            if st.button("Iniciar Detección", type="primary"):
                if st.session_state.user_id is None:
                    st.warning("Selecciona o registra un usuario antes de iniciar")
                else:
                    st.session_state.session_active = True
                    st.session_state.start_time = datetime.now()
                    st.session_state.emotion_data = pd.DataFrame()
                    st.session_state.frame_count = 0
                    st.session_state.show_report = False
                    st.session_state.last_detected_emotions = None
                    # Create session in DB
                    try:
                        session_db_id = db.create_session(st.session_state.user_id, st.session_state.start_time)
                        st.session_state.session_id = session_db_id
                    except Exception as e:
                        st.error(f"Error creando sesión en la base de datos: {e}")
                        st.session_state.session_active = False
                    st.rerun()
        else:
            if st.button("Detener Sesión", type="secondary"):
                st.session_state.session_active = False
                st.session_state.show_report = True
                st.rerun()

        # UI placeholders
        if st.session_state.session_active:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Video en Vivo")
                video_placeholder = st.empty()
            with col2:
                st.subheader("Emociones Actuales")
                bars_placeholder = st.empty()

            st.markdown("---")
            graph_placeholder = st.empty()

            # Capture
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            try:
                while st.session_state.session_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("No se pudo acceder a la cámara")
                        break

                    st.session_state.frame_count += 1

                    if st.session_state.frame_count % 2 == 0:
                        emotions, box = detect_emotions_from_frame(frame)

                        if emotions and box:
                            x, y, w, h = box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            st.session_state.last_detected_emotions = emotions

                            if st.session_state.frame_count % 10 == 0:
                                elapsed = (datetime.now() - st.session_state.start_time).total_seconds()
                                new_row = create_emotion_dataframe(emotions, datetime.now())
                                new_row['time_seconds'] = elapsed
                                st.session_state.emotion_data = pd.concat([st.session_state.emotion_data, new_row], ignore_index=True)

                                # Update visuals
                                bars_placeholder.plotly_chart(create_emotion_bars(emotions), use_column_width=True)
                                if len(st.session_state.emotion_data) > 1:
                                    graph_placeholder.plotly_chart(create_realtime_graph(st.session_state.emotion_data), use_column_width=True)

                                # Save to DB
                                try:
                                    db.save_emotion_record(st.session_state.session_id, elapsed, emotions)
                                except Exception as e:
                                    st.warning(f"No se guardó un registro en la BDD: {e}")

                        else:
                            # No face detected
                            if st.session_state.frame_count % 10 == 0:
                                elapsed = (datetime.now() - st.session_state.start_time).total_seconds()
                                zero_emotions = {emotion: 0 for emotion in EMOTIONS_ORDER}
                                new_row = create_emotion_dataframe(zero_emotions, datetime.now())
                                new_row['time_seconds'] = elapsed
                                st.session_state.emotion_data = pd.concat([st.session_state.emotion_data, new_row], ignore_index=True)

                                bars_placeholder.plotly_chart(create_emotion_bars(zero_emotions), use_column_width=True)
                                if len(st.session_state.emotion_data) > 1:
                                    graph_placeholder.plotly_chart(create_realtime_graph(st.session_state.emotion_data), use_column_width=True)

                                try:
                                    db.save_emotion_record(st.session_state.session_id, elapsed, zero_emotions)
                                except Exception as e:
                                    st.warning(f"No se guardó un registro en la BDD: {e}")

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    time.sleep(0.03)

            finally:
                cap.release()

        elif st.session_state.show_report and not st.session_state.emotion_data.empty:
            st.success("Sesión finalizada")
            report_data = generate_report_data(
                st.session_state.emotion_data,
                st.session_state.session_id if st.session_state.session_id else random.randint(100, 999),
                st.session_state.start_time,
                datetime.now()
            )

            display_session_report(report_data, st.session_state.emotion_data)
            
            # Save final info to DB
            try:
                db.finish_session(
                    st.session_state.session_id,
                    report_data["end_time"],
                    int(report_data["duration"].total_seconds()),
                    report_data["total_frames"]
                )
  
                #    Necesitamos convertir los objetos 'datetime' y 'timedelta'
                report_data_para_json = report_data.copy()

                # Convertir 'datetime' a string (formato ISO)
                report_data_para_json['start_time'] = report_data_para_json['start_time'].isoformat()
                report_data_para_json['end_time'] = report_data_para_json['end_time'].isoformat()
                
                # Convertir 'timedelta' a segundos (número)
                report_data_para_json['duration'] = report_data_para_json['duration'].total_seconds()
                
                # 3. Guardar el reporte JSON con los datos ya convertidos
                db.save_session_report(st.session_state.session_id, report_data_para_json)
            
            except Exception as e:
                st.warning(f"No se pudo guardar la información final en la BDD: {e}")

            if st.button("🔄 Nueva Sesión", type="primary"):
                st.session_state.show_report = False
                st.session_state.session_id = None
                st.session_state.emotion_data = pd.DataFrame()
                st.session_state.frame_count = 0
                st.rerun()

        else:
            st.info("Haz clic en 'Iniciar Detección' en el panel lateral para comenzar")

    else:
        # File upload mode: keep original behavior
        st.subheader("Analizar imagen o video")
        uploaded_file = st.file_uploader("Subir imagen o video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
        if uploaded_file:
            file_type = uploaded_file.type
            if 'image' in file_type:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                emotions, box = detect_emotions_from_frame(img)
                col1, col2 = st.columns([2, 1])
                with col1:
                    if emotions and box:
                        x, y, w, h = box
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="Imagen Analizada", use_column_width=True)
                with col2:
                    if emotions:
                        st.plotly_chart(create_emotion_bars(emotions), use_column_width=True)
                    else:
                        st.warning("No se detectaron rostros en la imagen")
            elif 'video' in file_type:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                vf = cv2.VideoCapture(tfile.name)
                
                st.markdown("### Análisis de Video")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    video_placeholder = st.empty()
                with col2:
                    bars_placeholder = st.empty()
                
                graph_placeholder = st.empty()
                
                # Datos para el reporte final
                emotion_data_video = pd.DataFrame()
                frame_count = 0
                
                # OPTIMIZACIÓN 1: Configurar salto de frames
                # Si notas que sigue lento, aumenta este número (ej. a 10 o 15)
                FRAME_SKIP = 5 
                
                # Obtener info del video para la barra de progreso
                total_frames_video = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_video = vf.get(cv2.CAP_PROP_FPS)
                if fps_video <= 0: fps_video = 30 # Fallback por si falla la lectura de FPS
                
                progress_bar = st.progress(0)
                
                start_process_time = datetime.now() # Para calcular cuánto tardamos nosotros

                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Solo procesamos cada N frames
                    if frame_count % FRAME_SKIP == 0:
                        
                        # OPTIMIZACIÓN 2: Redimensionar la imagen para detección
                        # Esto hace que el análisis sea MUCHO más rápido si el video es HD/4K
                        height, width = frame.shape[:2]
                        target_width = 640
                        scaling_factor = target_width / float(width)
                        
                        # Si el video ya es pequeño, no lo cambiamos
                        if scaling_factor < 1:
                            new_dims = (target_width, int(height * scaling_factor))
                            frame_small = cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)
                        else:
                            frame_small = frame
                            scaling_factor = 1

                        # Actualizar barra
                        if total_frames_video > 0:
                            progress_bar.progress(min(frame_count / total_frames_video, 1.0))

                        # Detectar en la imagen pequeña
                        emotions, box = detect_emotions_from_frame(frame_small)
                        
                        if emotions and box:
                            # Dibujar rectángulo
                            x, y, w, h = box
                            cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Calcular tiempo "simulado" del video (no el tiempo real de procesamiento)
                            video_timestamp_seconds = frame_count / fps_video
                            
                            new_row = create_emotion_dataframe(emotions, datetime.now()) # Timestamp real para referencia
                            new_row['time_seconds'] = video_timestamp_seconds # Timestamp del video para la gráfica
                            emotion_data_video = pd.concat([emotion_data_video, new_row], ignore_index=True)
                            
                            # Actualizar gráficas
                            bars_placeholder.plotly_chart(create_emotion_bars(emotions), use_column_width=True)
                            if len(emotion_data_video) > 1:
                                graph_placeholder.plotly_chart(create_realtime_graph(emotion_data_video), use_column_width=True)
                        
                        # Mostrar el frame (usamos el pequeño para ahorrar ancho de banda y CPU)
                        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                vf.release()
                progress_bar.empty() # Quitar la barra al terminar
                
                # --- GENERAR EL INFORME FINAL ---
                if not emotion_data_video.empty:
                    st.success("Análisis de video completado")
                    st.markdown("---")
                    
                    # Simulamos tiempos de inicio/fin basados en la duración del video
                    duration_sec = frame_count / fps_video
                    video_start = datetime.now() - timedelta(seconds=duration_sec)
                    video_end = datetime.now()
                    
                    # Generamos datos del reporte
                    report_data = generate_report_data(
                        emotion_data_video,
                        "--", # ID ficticio
                        video_start,
                        video_end
                    )
                    
                    # Sobrescribimos la duración calculada con la duración real del video analizado
                    report_data['duration'] = timedelta(seconds=duration_sec)

                    # ¡Llamamos a tu función mágica!
                    display_session_report(report_data, emotion_data_video)
                    
                else:
                    st.warning("No se detectaron rostros o emociones en el video.")

# ---------------------------
# Page: HISTORIAL (view previous sessions)
# ---------------------------
elif st.session_state.page == "Historial":
    st.subheader("Historial de Sesiones")
    if st.session_state.user_id is None:
        st.info("Selecciona un usuario en la barra lateral para ver su historial")
    else:
        try:
            sessions = db.get_sessions_by_user(st.session_state.user_id)
        except Exception as e:
            sessions = []
            st.error(f"Error obteniendo sesiones: {e}")

        if not sessions:
            st.info("No hay sesiones registradas para este usuario")
        else:
            session_options = [f"{s['id']} ({s['start_time']})" for s in sessions]
            
            st.markdown("---")
            sel_session_str = st.selectbox("Selecciona sesión para cargar", ["-- Ninguna --"] + session_options)
            
            if sel_session_str != "-- Ninguna --":
                sid = None
                try:
                    # 3. Extraemos el ID del string (ej: "3 (2025-...)")
                    sid = int(sel_session_str.split(" ")[0])
                except (ValueError, IndexError):
                    st.error("Error al leer el ID de la sesión.")

                if sid:
                    # (El resto de tu lógica para cargar reportes sigue aquí...)
                    try:
                        records = db.get_emotion_records(sid)
                        records_df = pd.DataFrame(records)
                        
                        if not records_df.empty:
                            if 'time_seconds' not in records_df.columns:
                                records_df['time_seconds'] = range(len(records_df))
                            
                            # Mostrar reporte si existe
                            try:
                                report = db.get_report(sid)
                            except Exception as e:
                                report = None
                                st.warning(f"No se pudo cargar reporte desde BDD: {e}")

                            if report:
                                display_session_report(report, records_df)
                            else:
                                # Generar reporte "on the fly"
                                st.info("No se encontró reporte guardado, generando uno nuevo...")
                                generated = generate_report_data(records_df, sid, datetime.now(), datetime.now()) # Nota: las fechas pueden no ser precisas
                                if generated:
                                    display_session_report(generated, records_df)
                        else:
                            st.warning("No hay registros emocionales para esta sesión")
                    except Exception as e:
                        st.error(f"Error cargando registros de la sesión: {e}")

# ---------------------------
# Page: ADMINISTRAR (delete patients / sessions)
# ---------------------------
elif st.session_state.page == "Administrar":
    st.subheader("Administrar Usuarios y Sesiones")
    st.markdown("**Eliminar sesión**")
    try:
        patients = db.get_users()
    except Exception as e:
        patients = []
        st.error("Error cargando Usuarios")

    patient_map = {p["name"]: p["id"] for p in patients}
    sel_patient_for_admin = st.selectbox("Selecciona usuario", ["-- Ninguno --"] + list(patient_map.keys()))
    if sel_patient_for_admin != "-- Ninguno --":
        uid = patient_map[sel_patient_for_admin]
        try:
            sessions = db.get_sessions_by_user(uid)
        except Exception as e:
            sessions = []
            st.error("Error cargando sesiones del usuario")
        if sessions:
            sel_session = st.selectbox("Selecciona sesión a eliminar", ["-- Ninguna --"] + [str(s['id']) + " | " + str(s.get('start_time')) for s in sessions])
            if sel_session != "-- Ninguna --":
                sid = int(sel_session.split("|")[0].strip())
                if st.button("Eliminar sesión seleccionada"):
                    try:
                        if hasattr(db, "delete_session"):
                            db.delete_session(sid)
                            st.success("Sesión eliminada")
                            st.rerun()
                        else:
                            st.error("La función delete_session no existe en db.py")
                    except Exception as e:
                        st.error(f"Error eliminando sesión: {e}")
        else:
            st.info("El usuario no tiene sesiones")

    st.markdown("---")
    st.markdown("**Eliminar usuario**")
    sel_patient_del = st.selectbox("Selecciona usuario a eliminar", ["-- Ninguno --"] + list(patient_map.keys()))
    if sel_patient_del != "-- Ninguno --":
        uid_del = patient_map[sel_patient_del]
        st.warning("Eliminar un usuario también va a eliminar sus sesiones asociadas")
        if st.button("Eliminar usuario seleccionado"):
            try:
                if hasattr(db, "delete_user"):
                    db.delete_user(uid_del)
                    st.success("Usuario eliminado")
                    st.rerun()
                else:
                    st.error("La función delete_user no existe en db.py")
            except Exception as e:
                st.error(f"Error eliminando usuario: {e}")

# End of file
