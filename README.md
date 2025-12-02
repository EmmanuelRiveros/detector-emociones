# Aplicación de reconocimiento de emociones

Aplicación de reconocimiento de emociones basada en reconocimiento facial y análisis utilizando Python, Streamlit y OpenCV.

## Requisitos Previos

*   Python 3.9 o superior
*   MySQL Server
*   Git

## Instalación

1.  **Clonar el repositorio**

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    ```

2.  **Crear un entorno virtual**

    Es recomendable usar un entorno virtual para aislar las dependencias.

    ```bash
    python -m venv emociones-env
    ```

    *   En Windows:
        ```bash
        .\emociones-env\Scripts\activate
        ```
    *   En macOS/Linux:
        ```bash
        source emociones-env/bin/activate
        ```

3.  **Instalar dependencias**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar la Base de Datos**

    *   Asegúrate de tener MySQL corriendo.
    *   Crea una base de datos llamada `emotion_therapy` (o el nombre que prefieras).
    *   Importa el esquema de la base de datos desde el archivo `db/therapy.sql`.

    ```bash
    mysql -u root -p emotion_therapy < db/therapy.sql
    ```

5.  **Configurar Variables de Entorno**

    Crea un archivo `.env` en la raíz del proyecto basándote en la configuración de tu base de datos local:

    ```env
    DB_HOST=localhost
    DB_USER=tu_usuario
    DB_PASSWORD=tu_contraseña
    DB_NAME=emotion_therapy
    ```

## Ejecución

Para iniciar la aplicación, ejecuta el siguiente comando:

```bash
streamlit run app.py
```

## Notas Adicionales

*   **Reportes**: Los reportes de las sesiones se guardan en la carpeta `emotion_reports`.
