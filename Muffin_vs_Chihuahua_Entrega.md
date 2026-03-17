# Documento de Entrega — Vision Challenge

## Clasificación de imágenes: Chihuahua vs Muffin

**Asignatura:** Programación de Inteligencia Artificial  
**Unidad:** Deep Learning y Visión por Computador  
**Actividad:** Vision Challenge  
**Tipo de trabajo:** Trabajo en equipo  

---

# 1. Información del equipo

| Campo                  | Información                        |
| ---------------------- | ---------------------------------- |
| Nombre del equipo      | Madalenos                          |
| Integrantes            | Manuel Garrido Serrano, Daniel Mera Sachse, Israel Soto Cala, Juan Manuel Vega Carrillo |
| Fecha de entrega       | 17 de marzo de 2026               |
| Arquitectura utilizada | CNN (4 bloques convolucionales)    |
| Presentación           | `entrega/Chihuahua vs Muffin.pptx` |

---

# 2. Objetivo del proyecto

El objetivo de esta actividad es desarrollar un modelo de **visión artificial** capaz de distinguir entre imágenes de:

* **Chihuahuas**
* **Muffins**

Se trata de un problema de **clasificación binaria de imágenes** utilizando técnicas de **Deep Learning**. El reto es especialmente interesante porque ambas clases comparten patrones visuales similares (forma redonda, tonos marrones, texturas), lo que lo convierte en un clásico benchmark de visión por computador.

El equipo ha:

* diseñado un modelo CNN con técnicas de regularización avanzadas
* entrenado con data augmentation para mejorar la generalización
* evaluado el rendimiento con múltiples métricas (accuracy, precision, recall, F1, AUC-ROC)
* analizado los errores con Grad-CAM para interpretar las decisiones del modelo
* desarrollado una interfaz web interactiva con Gradio para demostración en vivo

---

# 3. Dataset utilizado

El dataset utilizado proviene de Kaggle (*Muffin vs Chihuahua Image Classification* de Samuel Cortinhas). Se preparó un subconjunto adecuado para el aula con el script `prepare_dataset_chihuahua_muffin.py`.

| Elemento                 | Descripción                                    |
| ------------------------ | ---------------------------------------------- |
| Número total de imágenes | ~1.000 (400 train + 100 test por clase)        |
| Clases                   | Chihuahua (0) / Muffin (1)                     |
| Tamaño de imagen         | Redimensionadas a 128×128 píxeles              |
| División del dataset     | train (~720) / validation (~80) / test (200)   |

La partición de validación se obtiene automáticamente desde el conjunto de train usando `validation_split=0.10` con seed fija (42) para reproducibilidad.

```
Train:      ~720 imágenes (90% del train original)
Validation: ~80 imágenes  (10% del train original)
Test:       200 imágenes  (100 chihuahuas + 100 muffins)
```

---

# 4. Preprocesado de datos

Se aplicaron las siguientes transformaciones:

**Normalización:**
* Escalado de píxeles de [0, 255] a [0, 1] mediante `Rescaling(1./255)` integrado como primera capa del modelo.

**Data Augmentation** (integrado en el modelo, activo solo en entrenamiento):
* `RandomFlip("horizontal")` — Espejado horizontal aleatorio
* `RandomRotation(0.15)` — Rotación aleatoria hasta ±15%
* `RandomZoom(0.15)` — Zoom aleatorio hasta ±15%
* `RandomContrast(0.1)` — Variación de contraste hasta ±10%

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
])
```

**Optimización de carga:**
* `cache()` y `prefetch(AUTOTUNE)` para acelerar el pipeline de datos durante el entrenamiento.

---

# 5. Arquitectura del modelo

Se diseñó una **CNN Sequential** con 4 bloques convolucionales de filtros crecientes, Batch Normalization y Dropout:

```
Input (128×128×3)
├── Rescaling(1/255)           → Normalización
├── Data Augmentation          → Solo en entrenamiento
│
├── Bloque 1: Conv2D(32, 3×3, relu, padding=same) → BatchNorm → MaxPool(2×2)
├── Bloque 2: Conv2D(64, 3×3, relu, padding=same) → BatchNorm → MaxPool(2×2)
├── Bloque 3: Conv2D(128, 3×3, relu, padding=same) → BatchNorm → MaxPool(2×2)
├── Bloque 4: Conv2D(256, 3×3, relu, padding=same) → BatchNorm → MaxPool(2×2)
│
├── Flatten
├── Dense(256, relu) → BatchNorm → Dropout(0.5)
├── Dense(128, relu) → Dropout(0.3)
└── Dense(1, sigmoid)          → Salida binaria
```

**Decisiones de diseño:**
* **Batch Normalization** después de cada convolución para estabilizar el entrenamiento y permitir learning rates más altos.
* **Dropout agresivo** (0.5 y 0.3) en las capas densas para combatir el overfitting en un dataset pequeño.
* **Padding='same'** para preservar dimensiones espaciales dentro de cada bloque.
* **Filtros crecientes** (32→64→128→256) para capturar características de complejidad progresiva.

---

# 6. Entrenamiento del modelo

| Parámetro         | Valor                        |
| ----------------- | ---------------------------- |
| Optimizer         | Adam                         |
| Loss              | binary_crossentropy          |
| Batch size        | 32                           |
| Epochs (máx.)     | 30 (entrenó 12)              |
| Learning rate     | 0.001                        |
| Tiempo máx.       | 10 minutos (TimeStopping)    |

**Callbacks utilizados:**
* **EarlyStopping:** monitoriza `val_loss`, patience=5, restaura mejores pesos
* **ReduceLROnPlateau:** reduce el learning rate ×0.5 si val_loss se estanca 3 épocas, mínimo 1e-6
* **TimeStopping:** detiene el entrenamiento si supera 10 minutos
* **TensorBoard:** logs de métricas e histogramas por época

---

# 7. Resultados obtenidos

| Métrica                  | Valor         |
| ------------------------ | ------------- |
| Accuracy entrenamiento   | 93.07%        |
| Accuracy validación      | 90.44%        |
| **Accuracy test**        | **91.45%**    |
| Loss entrenamiento       | 0.1793        |
| Loss validación (mejor)  | 0.2322        |
| **AUC-ROC (test)**       | **0.972**     |
| **AUC-PR (test)**        | **0.974**     |

**Evolución del entrenamiento (12 épocas):**

| Época | Train Acc | Val Acc | Train Loss | Val Loss |
| ----- | --------- | ------- | ---------- | -------- |
| 1     | 78.29%    | 50.35%  | 0.5223     | 1.6817   |
| 2     | 83.80%    | 50.35%  | 0.4124     | 3.1950   |
| 3     | 85.33%    | 55.94%  | 0.3682     | 1.1971   |
| 4     | 87.37%    | 86.48%  | 0.3138     | 0.3134   |
| 5     | 88.80%    | 87.88%  | 0.2785     | 0.3001   |
| 6     | 89.34%    | 90.44%  | 0.2642     | 0.2580   |
| 7     | 90.30%    | 85.55%  | 0.2467     | 0.3253   |
| 8     | 90.76%    | 90.44%  | 0.2278     | 0.2418   |
| 9     | 91.33%    | 88.58%  | 0.2124     | 0.2447   |
| 10    | 91.82%    | 89.98%  | 0.1983     | 0.2322   |
| 11    | 92.34%    | 90.44%  | 0.1942     | 0.2376   |
| 12    | 93.07%    | 81.59%  | 0.1793     | 0.3899   |

**Observaciones:**
* El modelo converge rápidamente (época 4 ya supera el 86% en validación).
* La mejor val_accuracy (90.44%) se alcanza en las épocas 6, 8 y 11.
* En la época 12 la val_accuracy baja a 81.59% indicando leve overfitting; el EarlyStopping restauraría los mejores pesos.
* El modelo generaliza bien: **accuracy en test (91.45%) ≥ accuracy en validación (90.44%)**.
* Las curvas de entrenamiento, la matriz de confusión y el análisis completo están en el notebook de Israel (Persona C).

**Matriz de confusión (test — 1.076 imágenes):**

| Real \ Predicho | Chihuahua | Muffin |
| --------------- | --------- | ------ |
| Chihuahua (534) | **488**   | 46     |
| Muffin (542)    | 46        | **496**|

- Errores totales: 92 de 1.076 (8.55%)
- El modelo comete exactamente el mismo número de errores en ambas clases → está bien equilibrado

**Métricas detalladas por clase:**

| Clase      | Precision | Recall | F1     |
| ---------- | --------- | ------ | ------ |
| Chihuahua  | 91.39%    | 91.39% | 91.39% |
| Muffin     | 91.51%    | 91.51% | 91.51% |
| **Macro**  | **91.45%**|**91.45%**|**91.45%**|

---

# 8. Experimentos realizados

## Baseline vs Modelo Mejorado

Se partió de un **modelo baseline** proporcionado en `notebook_base_chihuahua_vs_muffin.ipynb`:
* 3 capas Conv2D (32, 64, 128) sin BatchNorm ni Dropout
* Sin Data Augmentation
* 5 épocas con Adam por defecto

**Mejoras implementadas respecto al baseline:**

| Técnica              | Baseline | Modelo final | Impacto                                         |
| -------------------- | -------- | ------------ | ----------------------------------------------- |
| Data Augmentation    | No       | Sí           | Mejora generalización, reduce overfitting        |
| Batch Normalization  | No       | Sí           | Entrenamiento más estable                        |
| Dropout (0.5/0.3)    | No       | Sí           | Regularización de capas densas                   |
| Bloques Conv2D       | 3        | 4            | Mayor capacidad de extracción de características |
| Filtros máximos      | 128      | 256          | Mejor representación de alto nivel               |
| EarlyStopping        | No       | Sí           | Evita sobreentrenamiento                         |
| ReduceLROnPlateau    | No       | Sí           | Ajuste fino del learning rate                    |

**Qué funcionó mejor:**
* Data Augmentation fue la técnica que más impactó en la generalización.
* Batch Normalization permitió entrenar más rápido y estable.
* El Dropout agresivo (0.5) fue clave para evitar overfitting severo en un dataset pequeño.

**Qué no funcionó tan bien:**
* Las primeras 2-3 épocas muestran inestabilidad en la validación (val_loss muy alto), antes de estabilizarse.
* En la última época (12) se observa leve overfitting.

---

# 9. Análisis de errores

El notebook de Persona C (`notebook_persona_c_visualizacion_errores_tensorboard.ipynb`) realiza un análisis detallado de errores que incluye:

**Métricas calculadas:**
* Accuracy, Precision, Recall, F1-score (binario y macro)
* AUC-ROC y curva Precision-Recall
* Matriz de confusión (conteos y normalizada)

**Tipos de errores detectados (92 errores en 1.076 imágenes):**
* **46 chihuahuas → clasificados como muffin:** el modelo confunde chihuahuas pequeños, de color claro y forma redondeada con muffins. También falla en imágenes artísticas/dibujadas de chihuahuas con aspecto inusual.
* **46 muffins → clasificados como chihuahua:** el modelo confunde muffins en contextos inusuales (ingredientes a su alrededor, ángulos extraños, fondos complejos).

**Imágenes más confusas:** imágenes con fondo muy elaborado, ángulos extremos, ilustraciones artísticas, o fotos de muy baja calidad.

**AUC-ROC = 0.972 · AUC-PR = 0.974** — el modelo discrimina excepcionalmente bien entre clases, incluso cuando sus predicciones exactas no son perfectas.

**Grad-CAM (interpretabilidad):**
* Se generaron mapas de activación que muestran qué regiones de la imagen influyeron en la predicción.
* En las predicciones correctas, el modelo se centra en rasgos faciales (ojos, nariz) para chihuahuas y en la textura/forma circular para muffins.
* En los errores, el modelo se distrae con texturas y bordes ambiguos.

**Cómo podría mejorarse:**
* Transfer learning con un modelo preentrenado (ej. MobileNetV2) podría mejorar la extracción de características.
* Aumentar el tamaño del dataset con más ejemplos difíciles.
* Usar mixup o cutmix como data augmentation avanzada.
* Ajustar el umbral de confianza para los casos más ambiguos.

---

# 10. Conclusiones

**Principales dificultades:**
* La similitud visual entre chihuahuas y muffins (forma redonda, tonos marrones) hace que el problema sea genuinamente difícil.
* Estabilizar el entrenamiento en las primeras épocas requirió ajustar data augmentation y BatchNorm.
* Con ~4.300 imágenes de entrenamiento, la regularización agresiva fue esencial para evitar overfitting.

**Mejoras posibles:**
* Transfer learning con redes preentrenadas (MobileNetV2, EfficientNet).
* Ensemble de varios modelos para reducir la varianza de las predicciones.
* Implementar test-time augmentation (TTA) para mejorar la confianza.
* Probar Vision Transformers (ViT) pequeños para comparar con la CNN.

**Qué hemos aprendido:**
* La importancia del Data Augmentation en datasets pequeños.
* Cómo usar Batch Normalization y Dropout juntos para regularizar.
* La utilidad de herramientas como Grad-CAM y TensorBoard para entender y depurar un modelo.
* Cómo desplegar un modelo con Gradio para crear demos interactivas.
* El valor del trabajo en equipo y la especialización de roles.

---

# 11. Contribución de cada miembro

| Estudiante                  | Contribución                                           |
| --------------------------- | ------------------------------------------------------ |
| Juan Manuel Vega Carrillo   | Preparación del dataset, preprocesado y data pipeline  |
| Manuel Garrido Serrano      | Diseño de arquitectura CNN y entrenamiento del modelo  |
| Israel Soto Cala            | Visualización de errores, Grad-CAM y TensorBoard       |
| Daniel Mera Sachse          | Documentación, entrega y defensa del proyecto          |
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

