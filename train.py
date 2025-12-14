import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------
# Configuración
# ---------------------------------------------------------
IMG_width, IMG_height = 48, 48 
BATCH_SIZE = 32
EPOCHS = 1
NUM_CLASSES = 7 
LEARNING_RATE = 0.001

TRAIN_DIR = 'data/train' 
VAL_DIR = 'data/test'

def build_model():
    """
    Construye el modelo con las 3 modificaciones solicitadas:
    1. Base pre-entrenada (Transfer Learning)
    2. Capas Densas Personalizadas (Arquitectura)
    3. Dropout (Regularización)
    """
    # Usamos MobileNetV2 como base (Transfer Learning)
    # include_top=False significa que quitamos las capas de clasificación originales
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_width, IMG_height, 3))
    
    # Congelamos las capas base para no dañar los pesos pre-entrenados al inicio
    base_model.trainable = False 

    # Construimos la "cabeza" del modelo
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # --- MODIFICACIÓN: ARQUITECTURA (Capa Densa) ---
    # Añadimos una capa densa con 256 neuronas como solicitado
    x = Dense(256, activation='relu')(x)
    
    # --- MODIFICACIÓN: REGULARIZACIÓN (Dropout) ---
    # Añadimos Dropout del 50% para evitar overfitting
    x = Dropout(0.5)(x)
    
    # Capa de salida
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def main():
    # ---------------------------------------------------------
    # 1. MODIFICACIÓN: AUMENTO DE DATOS (Data Augmentation)
    # ---------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,      # Rotación aleatoria
        width_shift_range=0.2,  # Desplazamiento horizontal
        height_shift_range=0.2, # Desplazamiento vertical
        shear_range=0.2,        # Inclinación
        zoom_range=0.2,         # Zoom aleatorio
        horizontal_flip=True,   # Espejo horizontal
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_width, IMG_height),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_width, IMG_height),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # ---------------------------------------------------------
    # Construir y Compilar Modelo
    # ---------------------------------------------------------
    model = build_model()
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # ---------------------------------------------------------
    # Entrenar
    # ---------------------------------------------------------
    print("Iniciando entrenamiento...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Guardar el modelo
    model.save('modelo_emociones_custom.h5')
    print("Modelo guardado como 'modelo_emociones_custom.h5'")

if __name__ == '__main__':
    main()
