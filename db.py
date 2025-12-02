import mysql.connector
import json
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def create_user(name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name) VALUES (%s)", (name,))
    conn.commit()
    user_id = cur.lastrowid
    cur.close()
    conn.close()
    return user_id

def get_users():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()
    conn.close()
    return users

def create_session(user_id, start_time):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sessions (user_id, start_time)
        VALUES (%s, %s)
    """, (user_id, start_time))
    conn.commit()
    session_id = cur.lastrowid
    cur.close()
    conn.close()
    return session_id

def save_emotion_record(session_id, time_seconds, emotions):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO emotion_records 
        (session_id, time_seconds, happy, sad, angry, surprise, fear, disgust, neutral)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        session_id, time_seconds,
        emotions["happy"], emotions["sad"], emotions["angry"],
        emotions["surprise"], emotions["fear"], emotions["disgust"], emotions["neutral"]
    ))
    conn.commit()
    cur.close()
    conn.close()

def finish_session(session_id, end_time, duration, total_frames):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sessions
        SET end_time = %s, duration_seconds = %s, total_frames = %s
        WHERE id = %s
    """, (end_time, duration, total_frames, session_id))
    conn.commit()
    cur.close()
    conn.close()

def save_session_report(session_id, report_data):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO session_reports (session_id, report_json)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE report_json = VALUES(report_json)
    """, (session_id, json.dumps(report_data)))
    conn.commit()
    cur.close()
    conn.close()

def get_sessions_by_user(user_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM sessions WHERE user_id = %s ORDER BY start_time DESC", (user_id,))
    sessions = cur.fetchall()
    cur.close()
    conn.close()
    return sessions

def get_emotion_records(session_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM emotion_records WHERE session_id = %s", (session_id,))
    records = cur.fetchall()
    cur.close()
    conn.close()
    return records

def get_report(session_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT report_json FROM session_reports WHERE session_id = %s", (session_id,))
    report = cur.fetchone()
    cur.close()
    conn.close()
    return json.loads(report["report_json"]) if report else None

def delete_session(session_id):
    """Elimina una sesión y todos sus registros vinculados."""
    conn = get_connection()
    cur = conn.cursor()

    # Primero borrar registros emocionales
    cur.execute("DELETE FROM emotion_records WHERE session_id = %s", (session_id,))

    # Luego borrar reporte si existe
    cur.execute("DELETE FROM session_reports WHERE session_id = %s", (session_id,))

    # Finalmente borrar la sesión
    cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    conn.commit()
    cur.close()
    conn.close()
    return True


def delete_user(user_id):
    """Elimina un usuario y todas sus sesiones y datos relacionados."""
    conn = get_connection()
    cur = conn.cursor()

    # Obtener sesiones para borrarlas individualmente
    cur.execute("SELECT id FROM sessions WHERE user_id = %s", (user_id,))
    sessions = cur.fetchall()

    for session in sessions:
        sid = session[0]

        # Eliminar registros emocionales
        cur.execute("DELETE FROM emotion_records WHERE session_id = %s", (sid,))

        # Eliminar reportes
        cur.execute("DELETE FROM session_reports WHERE session_id = %s", (sid,))

    # Eliminar todas las sesiones del usuario
    cur.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))

    # Finalmente eliminar el usuario
    cur.execute("DELETE FROM users WHERE id = %s", (user_id,))

    conn.commit()
    cur.close()
    conn.close()
    return True
