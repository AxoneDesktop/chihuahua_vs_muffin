"""
Interfaz Gradio para clasificar imágenes: Chihuahua vs Muffin
Preparado para Hugging Face Spaces
"""

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
from PIL import Image

# ============================================================================
# 1. CARGAR MODELO (lazy para evitar conflicto con spaces/codefind)
# ============================================================================

MODEL_PATH = Path(__file__).parent / "modelo_chihuahua_vs_muffin.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Chihuahua", "Muffin"]
THRESHOLD = 0.5

tf = None
model = None

def _load_model():
    global tf, model
    if model is None:
        import tensorflow as _tf
        tf = _tf
        print(f"Cargando modelo desde: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo cargado correctamente")
    return tf, model

# ============================================================================
# 2. FUNCIONES DE PREDICCIÓN
# ============================================================================

def preprocess_input_image(image):
    """Preprocesa imagen: RGB 128x128 float32."""
    if isinstance(image, Image.Image):
        img_pil = image.convert("RGB").resize(IMG_SIZE)
    else:
        np_img = np.array(image)
        if np_img.ndim == 2:
            np_img = np.stack([np_img] * 3, axis=-1)
        elif np_img.ndim == 3 and np_img.shape[-1] == 4:
            np_img = np_img[:, :, :3]
        np_img = np.clip(np_img, 0, 255).astype("uint8")
        img_pil = Image.fromarray(np_img).convert("RGB").resize(IMG_SIZE)

    return np.array(img_pil).astype("float32")


def predict_image(image):
    """Predice si la imagen es chihuahua o muffin."""
    tf, model = _load_model()
    img_array = preprocess_input_image(image)
    input_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(input_array, verbose=0)[0][0]

    if prediction >= THRESHOLD:
        label = CLASS_NAMES[1]  # Muffin
        confidence = float(prediction)
    else:
        label = CLASS_NAMES[0]  # Chihuahua
        confidence = float(1 - prediction)

    probabilities = {
        CLASS_NAMES[0]: float(1 - prediction),
        CLASS_NAMES[1]: float(prediction),
    }

    return label, confidence, probabilities


def predict_with_output(image):
    """Función para la pestaña de predicción simple."""
    if image is None:
        return "Sube una imagen para clasificar.", {}

    label, confidence, probs = predict_image(image)

    result_text = (
        f"**RESULTADO: {label.upper()}**\n\n"
        f"Confianza: {confidence * 100:.1f}%\n\n---\n\n"
        f"Probabilidades:\n"
        f"- Chihuahua: {probs['Chihuahua'] * 100:.1f}%\n"
        f"- Muffin: {probs['Muffin'] * 100:.1f}%"
    )

    return result_text, probs


# ============================================================================
# 2b. GRAD-CAM
# ============================================================================

def get_last_conv_layer_name(mdl):
    """Obtiene nombre de la última capa convolucional."""
    tf, _ = _load_model()
    for layer in reversed(mdl.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No se encontró una capa Conv2D en el modelo.")


def forward_with_intermediate(mdl, inputs, target_layer_name, training=False):
    """Forward pass capa por capa para obtener activaciones intermedias."""
    x = inputs
    target_activation = None

    for layer in mdl.layers:
        try:
            x = layer(x, training=training)
        except TypeError:
            x = layer(x)
        if layer.name == target_layer_name:
            target_activation = x

    if target_activation is None:
        raise ValueError(f"No se encontró la capa objetivo: {target_layer_name}")

    return target_activation, x


def make_gradcam_heatmap(img_array, mdl, last_conv_layer_name, pred_index=0):
    """Genera mapa de activación Grad-CAM."""
    tf, _ = _load_model()
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, preds = forward_with_intermediate(
            mdl, img_tensor, last_conv_layer_name, training=False
        )
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Superpone el mapa de calor sobre la imagen."""
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(
            (img.shape[1], img.shape[0])
        )
    )
    cmap = plt.get_cmap("jet")
    jet_colors = cmap(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = (jet_heatmap * 255).astype("uint8")

    superimposed = jet_heatmap * alpha + img
    return np.clip(superimposed, 0, 255).astype("uint8")


def predict_with_gradcam(image):
    """Predicción con visualización Grad-CAM."""
    if image is None:
        return "Sube una imagen para analizar.", None

    label, confidence, probs = predict_image(image)
    img_array = preprocess_input_image(image)

    try:
        tf, model = _load_model()
        input_array = np.expand_dims(img_array, axis=0)
        last_conv = get_last_conv_layer_name(model)

        heatmap = make_gradcam_heatmap(input_array, model, last_conv, pred_index=0)
        overlay = overlay_heatmap(img_array.astype("uint8"), heatmap, alpha=0.4)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_array.astype("uint8"))
        axes[0].set_title("Imagen Original")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM ({label})")
        axes[1].axis("off")

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        gradcam_image = Image.open(buf)

        result_text = (
            f"**RESULTADO: {label.upper()}**\n\n"
            f"Confianza: {confidence * 100:.1f}%\n\n---\n\n"
            f"Probabilidades:\n"
            f"- Chihuahua: {probs['Chihuahua'] * 100:.1f}%\n"
            f"- Muffin: {probs['Muffin'] * 100:.1f}%\n\n"
            f"*(Las zonas rojas/amarillas en Grad-CAM indican las partes de la imagen que más influyeron)*"
        )

        return result_text, gradcam_image

    except Exception as e:
        result_text = (
            f"**RESULTADO: {label.upper()}**\n\n"
            f"Confianza: {confidence * 100:.1f}%\n\n---\n\n"
            f"Probabilidades:\n"
            f"- Chihuahua: {probs['Chihuahua'] * 100:.1f}%\n"
            f"- Muffin: {probs['Muffin'] * 100:.1f}%\n\n"
            f"(Grad-CAM no disponible: {e})"
        )
        return result_text, None


# ============================================================================
# 3. INTERFAZ GRADIO
# ============================================================================

with gr.Blocks(title="Chihuahua vs Muffin") as demo:
    gr.Markdown(
        "# 🐕 Chihuahua vs Muffin Classifier 🧁\n\n"
        "Sube una imagen y el modelo predecirá si es un **Chihuahua** o un **Muffin**.\n\n"
        "> El modelo funciona mejor con imágenes similares a las de entrenamiento (128×128px)"
    )

    with gr.Tabs():
        with gr.TabItem("Predicción Simple"):
            gr.Markdown("### Modo rápido")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Sube una imagen",
                        type="pil",
                        sources=["upload", "webcam", "clipboard"],
                    )
                    submit_btn = gr.Button("🔍 Predecir", variant="primary")
                with gr.Column():
                    output_text = gr.Markdown(label="Resultado")
                    output_chart = gr.Label(label="Confianza por clase")

            submit_btn.click(
                predict_with_output,
                inputs=input_image,
                outputs=[output_text, output_chart],
            )

        with gr.TabItem("Con Interpretabilidad (Grad-CAM)"):
            gr.Markdown(
                "### Modo avanzado – Mapa de activación\n\n"
                "Visualiza qué partes de la imagen influyeron más en la predicción."
            )
            with gr.Row():
                with gr.Column():
                    input_image_gradcam = gr.Image(
                        label="Sube una imagen",
                        type="pil",
                        sources=["upload", "webcam", "clipboard"],
                    )
                    submit_btn_gradcam = gr.Button("🔍 Analizar", variant="primary")
                with gr.Column():
                    output_text_gradcam = gr.Markdown(label="Resultado")

            output_gradcam = gr.Image(label="Análisis Grad-CAM", type="pil")

            submit_btn_gradcam.click(
                predict_with_gradcam,
                inputs=input_image_gradcam,
                outputs=[output_text_gradcam, output_gradcam],
            )

    gr.Markdown(
        "---\n\n"
        "**Modelo:** CNN entrenada con imágenes de chihuahuas y muffins · "
        "**Interpretabilidad:** Grad-CAM sobre la última capa convolucional"
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
