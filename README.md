# Aplicación de Reconocimiento de Emociones con Modelo Personalizado

Aplicación de reconocimiento de emociones en tiempo real utilizando **Streamlit**, **OpenCV** y **TensorFlow/Keras**. 

Esta versión ha sido optimizada para usar un **modelo personalizado (MobileNetV2)** entrenado con técnicas de *Data Augmentation* y *Regularización*, eliminando la dependencia de bases de datos para una configuración más sencilla.

## Características Nuevas

*   **Modelo Propio**: Utiliza `modelo_emociones_custom.h5`, un modelo de Deep Learning entrenado específicamente para este proyecto.
*   **Entrenamiento Flexible**: Incluye `train.py` para re-entrenar el modelo con tus propios datos.
*   **Detección Rápida**: Implementa *Haar Cascades* de OpenCV para una detección de rostros ágil en CPUs.
*   **Sin Base de Datos**: Funciona totalmente en memoria, sin necesidad de configurar MySQL.

## Requisitos Previos

*   Python 3.9 o superior
*   Git

## Instalación

1.  **Clonar el repositorio**

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd detector-emociones
    ```

2.  **Crear y activar un entorno virtual**

    ```bash
    python -m venv emociones-env
    ```

    *   **Windows**: `.\emociones-env\Scripts\activate`
    *   **macOS/Linux**: `source emociones-env/bin/activate`

3.  **Instalar dependencias**

    ```bash
    pip install -r requirements.txt
    ```

## Entrenamiento del Modelo

Si deseas entrenar tu propio modelo (o mejorar el existente), sigue estos pasos:

1.  **Preparar Datos**:
    Organiza tus imágenes en la carpeta `data/` con la siguiente estructura:
    ```
    data/
    ├── train/
    │   ├── angry/
    │   ├── happy/
    │   └── ... (7 emociones)
    └── test/
        ├── angry/
        └── ...
    ```

2.  **Ejecutar Entrenamiento**:
    
    *   **Opción A (Recomendada - Google Colab)**:
        1. Comprime la carpeta `data` y `train.py` en un zip.
        2. Súbelo a Google Colab.
        3. Ejecuta `python train.py` en un entorno con GPU.
        4. Descarga el archivo `modelo_emociones_custom.h5` generado.

    *   **Opción B (Local)**:
        Ejecuta el script directamente (puede tardar varias horas si no tienes GPU configurada):
        ```bash
        python train.py
        ```

3.  **Colocar el Modelo**:
    Asegúrate de que el archivo `modelo_emociones_custom.h5` esté en la raíz del proyecto.

## Ejecución

Para iniciar la aplicación:

```bash
streamlit run app.py
```

La aplicación cargará automáticamente `modelo_emociones_custom.h5` y comenzará la detección usando tu webcam.

## Estructura del Proyecto

*   `app.py`: Script principal de la aplicación Streamlit.
*   `train.py`: Script de entrenamiento con Keras/TensorFlow.
*   `modelo_emociones_custom.h5`: Archivo del modelo entrenado (debe estar presente para que la app funcione).
*   `requirements.txt`: Lista de dependencias.
