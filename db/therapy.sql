CREATE DATABASE IF NOT EXISTS emotion_therapy;
USE emotion_therapy;

-- Tabla de usuarios/pacientes
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de sesiones
CREATE TABLE sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    duration_seconds INT,
    total_frames INT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Línea de tiempo de emociones por sesión
CREATE TABLE emotion_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    time_seconds FLOAT NOT NULL,
    happy FLOAT,
    sad FLOAT,
    angry FLOAT,
    surprise FLOAT,
    fear FLOAT,
    disgust FLOAT,
    neutral FLOAT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Reportes procesados almacenados como JSON
CREATE TABLE session_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL UNIQUE,
    report_json JSON NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
