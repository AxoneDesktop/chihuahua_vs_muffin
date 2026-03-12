# Documento de Entrega — Vision Challenge

## Clasificación de imágenes: Chihuahua vs Muffin

**Asignatura:** Programación de Inteligencia Artificial
**Unidad:** Deep Learning y Visión por Computador
**Actividad:** Vision Challenge
**Tipo de trabajo:** Trabajo en equipo

---

# 1. Información del equipo

| Campo                  | Información         |
| ---------------------- | ------------------- |
| Nombre del equipo      |                     |
| Integrantes            |                     |
| Fecha de entrega       |                     |
| Arquitectura utilizada | CNN / ViT / híbrida |

---

# 2. Objetivo del proyecto

El objetivo de esta actividad es desarrollar un modelo de **visión artificial** capaz de distinguir entre imágenes de:

* **Chihuahuas**
* **Muffins**

Se trata de un problema de **clasificación binaria de imágenes** utilizando técnicas de **Deep Learning**.

El equipo debe:

* diseñar un modelo
* entrenarlo
* evaluar su rendimiento
* analizar sus resultados

---

# 3. Dataset utilizado

Describir brevemente el dataset.

**Información mínima:**

| Elemento                 | Descripción               |
| ------------------------ | ------------------------- |
| Número total de imágenes |                           |
| Clases                   | Chihuahua / Muffin        |
| Tamaño de imagen         |                           |
| División del dataset     | train / validation / test |

Ejemplo:

```
Train: 800 imágenes
Validation: 200 imágenes
Test: 200 imágenes
```

---

# 4. Preprocesado de datos

Describir las transformaciones aplicadas.

Ejemplos:

* redimensionado de imágenes
* normalización
* data augmentation
* balanceo de clases

Ejemplo de código:

```python
tf.keras.layers.Rescaling(1./255)
```

---

# 5. Arquitectura del modelo

Describir la arquitectura utilizada.

Ejemplo:

```
Input (128x128x3)
Conv2D 32
MaxPooling
Conv2D 64
MaxPooling
Conv2D 128
MaxPooling
Flatten
Dense 128
Dense 1 (sigmoid)
```

También se puede incluir:

* número de parámetros
* capas utilizadas
* técnicas de regularización

---

# 6. Entrenamiento del modelo

Describir los parámetros de entrenamiento.

| Parámetro     | Valor |
| ------------- | ----- |
| Optimizer     |       |
| Loss          |       |
| Batch size    |       |
| Epochs        |       |
| Learning rate |       |

Ejemplo:

```
optimizer = Adam
loss = binary_crossentropy
epochs = 10
batch_size = 32
```

---

# 7. Resultados obtenidos

Mostrar los resultados del modelo.

| Métrica                | Valor |
| ---------------------- | ----- |
| Accuracy entrenamiento |       |
| Accuracy validación    |       |
| Accuracy test          |       |

También incluir:

* curvas de entrenamiento
* ejemplos de predicción

---

# 8. Experimentos realizados

Describir qué pruebas se han hecho.

Ejemplos:

* cambio de número de capas
* uso de dropout
* data augmentation
* variación del learning rate

Explicar:

* qué funcionó mejor
* qué no funcionó

---

# 9. Análisis de errores

Mostrar ejemplos donde el modelo se equivoca.

Preguntas a responder:

* ¿Qué tipo de imágenes confunde el modelo?
* ¿Por qué puede ocurrir esto?
* ¿Cómo podría mejorarse?

---

# 10. Conclusiones

Breve reflexión del equipo:

* principales dificultades
* mejoras posibles
* qué se ha aprendido

---

# 11. Contribución de cada miembro

| Estudiante | Contribución |
| ---------- | ------------ |
|            |              |
|            |              |
|            |              |

Ejemplos:

* preparación del dataset
* diseño del modelo
* experimentación
* visualización de resultados
* documentación

---

# 12. Resultado para leaderboard

| Campo          | Valor |
| -------------- | ----- |
| Accuracy final |       |
| Arquitectura   |       |
| Observaciones  |       |

---

## Bonus (opcional)

Si el equipo ha implementado alguno de estos elementos:

* data augmentation avanzado
* visualización de activaciones
* comparación de arquitecturas
* mini Vision Transformer
* análisis detallado de errores

describirlo aquí.

