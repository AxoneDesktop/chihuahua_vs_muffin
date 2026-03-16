"""
Interfaz Gradio para clasificar imágenes: Chihuahua vs Muffin

Uso:
    python gradio_app.py

Luego abre en navegador: http://localhost:7860
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================================
# 1. CARGAR MODELO
# ============================================================================

# Detectar ruta del proyecto
CWD = Path.cwd().resolve()
project_candidates = [
    CWD,
    CWD.parent,
    CWD / 'chihuahua_vs_muffin',
    CWD.parent / 'chihuahua_vs_muffin',
    Path(r'C:\Users\usuario\Desktop\IABigData\repos\chihuahua_vs_muffin')
]

ROOT = None
for cand in project_candidates:
    if (cand / 'entrega').exists():
        ROOT = cand
        break

if ROOT is None:
    raise FileNotFoundError(
        f"No se encontró la carpeta del proyecto.\n"
        f"Intenta ejecutar desde: C:\\Users\\usuario\\Desktop\\IABigData\\repos\\chihuahua_vs_muffin"
    )

ENTREGA_DIR = ROOT / 'entrega'
MODEL_PATH = ENTREGA_DIR / 'modelo_chihuahua_vs_muffin.keras'

print(f"Cargando modelo desde: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (128, 128)
CLASS_NAMES = ['Chihuahua', 'Muffin']
THRESHOLD = 0.5

print("✅ Modelo cargado correctamente")

# ============================================================================
# 2. FUNCIONES DE PREDICCIÓN
# ============================================================================

def predict_image(image):
    """
    Predice si la imagen es chihuahua o muffin
    
    Args:
        image: imagen PIL o numpy array
        
    Returns:
        tuple: (label, confidence, {class_name: probability})
    """
    img_array = preprocess_input_image(image)
    input_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(input_array, verbose=0)[0][0]
    
    # Obtener etiqueta y confianza
    if prediction >= THRESHOLD:
        label = CLASS_NAMES[1]  # Muffin
        confidence = float(prediction)
    else:
        label = CLASS_NAMES[0]  # Chihuahua
        confidence = float(1 - prediction)
    
    # Diccionario con probabilidades
    probabilities = {
        CLASS_NAMES[0]: float(1 - prediction),
        CLASS_NAMES[1]: float(prediction)
    }
    
    return label, confidence, probabilities


def preprocess_input_image(image):
    """Preprocesa imagen como en entrenamiento: RGB 128x128 float32 en rango [0, 255]."""
    if hasattr(image, 'convert'):
        img_pil = image.convert('RGB').resize(IMG_SIZE)
        img_array = np.array(img_pil)
    else:
        from PIL import Image
        np_img = np.array(image)
        if np_img.ndim == 2:
            np_img = np.stack([np_img] * 3, axis=-1)
        elif np_img.ndim == 3 and np_img.shape[-1] == 4:
            np_img = np_img[:, :, :3]
        np_img = np.clip(np_img, 0, 255).astype('uint8')
        img_pil = Image.fromarray(np_img).convert('RGB').resize(IMG_SIZE)
        img_array = np.array(img_pil)

    return img_array.astype('float32')


def predict_with_output(image):
    """Función para Gradio con salida formateada"""
    label, confidence, probs = predict_image(image)
    
    # Crear mensaje de salida
    result_text = f"""
    **RESULTADO: {label.upper()}**
    
    Confianza: {confidence * 100:.1f}%
    
    ---
    
    Probabilidades:
    - Chihuahua: {probs['Chihuahua'] * 100:.1f}%
    - Muffin: {probs['Muffin'] * 100:.1f}%
    """
    
    return result_text, {CLASS_NAMES[0]: probs['Chihuahua'], CLASS_NAMES[1]: probs['Muffin']}


def get_last_conv_layer_name(model):
    """Obtiene nombre de la última capa convolucional"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError('No se encontró una capa Conv2D en el modelo.')


def forward_with_intermediate(model, inputs, target_layer_name, training=False):
    """Forward pass capa por capa para obtener activaciones intermedias"""
    x = inputs
    target_activation = None
    
    for layer in model.layers:
        try:
            x = layer(x, training=training)
        except TypeError:
            x = layer(x)
        
        if layer.name == target_layer_name:
            target_activation = x
    
    if target_activation is None:
        raise ValueError(f"No se encontró la capa objetivo: {target_layer_name}")
    
    return target_activation, x


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=0):
    """Genera mapa de activación Grad-CAM"""
    img_tensor = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = forward_with_intermediate(
            model, img_tensor, last_conv_layer_name, training=False
        )
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Superpone el mapa de calor sobre la imagen"""
    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = plt.get_cmap('jet')
    jet_colors = cmap(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    
    superimposed = jet_heatmap * alpha + img
    superimposed = np.clip(superimposed, 0, 255).astype('uint8')
    return superimposed


def predict_with_gradcam(image):
    """Predicción con visualización Grad-CAM"""
    # Obtener predicción
    label, confidence, probs = predict_image(image)
    
    # Preparar imagen con el mismo pipeline que la predicción normal
    img_array = preprocess_input_image(image)
    
    # Generar Grad-CAM
    try:
        input_array = np.expand_dims(img_array, axis=0)
        last_conv = get_last_conv_layer_name(model)
        
        # Índice de predicción
        pred_index = 1 if label == 'Muffin' else 0
        
        heatmap = make_gradcam_heatmap(input_array, model, last_conv, pred_index=pred_index)
        overlay = overlay_heatmap(img_array.astype('uint8'), heatmap, alpha=0.4)
        
        # Crear figura con imagen + Grad-CAM
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_array.astype('uint8'))
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        axes[1].imshow(overlay)
        axes[1].set_title(f'Grad-CAM ({label})')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Convertir a imagen para Gradio
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        import PIL
        gradcam_image = PIL.Image.open(buffer)
        
        result_text = f"""
        **RESULTADO: {label.upper()}**
        
        Confianza: {confidence * 100:.1f}%
        
        ---
        
        Probabilidades:
        - Chihuahua: {probs['Chihuahua'] * 100:.1f}%
        - Muffin: {probs['Muffin'] * 100:.1f}%
        
        *(Las zonas rojas/amarillas en Grad-CAM indican qué partes de la imagen influyeron más en la predicción)*
        """
        
        return result_text, gradcam_image
    
    except Exception as e:
        result_text = f"""
        **RESULTADO: {label.upper()}**
        
        Confianza: {confidence * 100:.1f}%
        
        ---
        
        Probabilidades:
        - Chihuahua: {probs['Chihuahua'] * 100:.1f}%
        - Muffin: {probs['Muffin'] * 100:.1f}%
        
        (Grad-CAM no disponible: {str(e)})
        """
        
        return result_text, None


# ============================================================================
# 3. INTERFAZ GRADIO
# ============================================================================

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🐕 Chihuahua vs Muffin Classifier 🧁
    
    Sube una imagen y el modelo predecirá si es un **Chihuahua** o un **Muffin**.
    
    > ⚠️ **Nota:** El modelo funciona mejor con imágenes de tamaño similar al de entrenamiento (128×128px)
    """)
    
    with gr.Tabs():
        # Tab 1: Predicción Simple
        with gr.TabItem("Predicción Simple"):
            gr.Markdown("### Modo rápido - Solo predicción")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Sube una imagen",
                        type="pil",
                        sources=["upload", "webcam", "clipboard"]
                    )
                    submit_btn = gr.Button("🔍 Predecir", variant="primary", scale=1)
                
                with gr.Column():
                    output_text = gr.Markdown(label="Resultado")
                    output_chart = gr.Label(
                        label="Confianza por clase",
                        show_label=True
                    )
            
            submit_btn.click(
                predict_with_output,
                inputs=input_image,
                outputs=[output_text, output_chart]
            )
        
        # Tab 2: Predicción con Grad-CAM
        with gr.TabItem("Con Interpretabilidad (Grad-CAM)"):
            gr.Markdown("""
            ### Modo avanzado - Con mapa de activación
            
            Visualiza qué partes de la imagen influyeron más en la predicción.
            """)
            
            with gr.Row():
                with gr.Column():
                    input_image_gradcam = gr.Image(
                        label="Sube una imagen",
                        type="pil",
                        sources=["upload", "webcam", "clipboard"]
                    )
                    submit_btn_gradcam = gr.Button("🔍 Analizar", variant="primary", scale=1)
                
                with gr.Column():
                    output_text_gradcam = gr.Markdown(label="Resultado")
            
            output_gradcam = gr.Image(
                label="Análisis Grad-CAM",
                type="pil"
            )
            
            submit_btn_gradcam.click(
                predict_with_gradcam,
                inputs=input_image_gradcam,
                outputs=[output_text_gradcam, output_gradcam]
            )
    
    gr.Markdown("""
    ---
    
    ### 📝 ¿Cómo funciona?
    
    **Síntesis:** El modelo es una red neuronal convolucional (CNN) entrenada con imágenes de chihuahuas y muffins.
    
    - **Predicción Simple:** Muestra la clase y su probabilidad
    - **Grad-CAM:** Visualiza las regiones que influyeron en la predicción mediante un mapa de calor
    
    | Rojo/Amarillo | = Regiones más importantes |
    | Azul | = Regiones menos importantes |
    """)

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🚀 Iniciando servidor Gradio...")
    print(f"{'='*70}")
    print(f"Abre en navegador: http://localhost:7860")
    print(f"Para detener: Ctrl + C en la terminal")
    print(f"{'='*70}\n")
    
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
