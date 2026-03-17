"""
Genera las gráficas de entrenamiento y el resumen del modelo.
No requiere el dataset, solo training_history.json y el modelo .keras.
Las imágenes se guardan en entrega/graficas_powerpoint/
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── Rutas ───────────────────────────────────────────────────────────────────
ENTREGA = Path(__file__).parent
HISTORY_PATH = ENTREGA / "training_history.json"
MODEL_PATH = ENTREGA / "modelo_chihuahua_vs_muffin.keras"
OUT_DIR = ENTREGA / "graficas_powerpoint"
OUT_DIR.mkdir(exist_ok=True)

# ─── Paleta consistente ───────────────────────────────────────────────────────
COLOR_TRAIN = "#2196F3"   # azul
COLOR_VAL   = "#FF5722"   # naranja
BEST_EPOCH  = 10          # 0-indexed → época 11 (val_loss mínimo = 0.2322)

# ─── 1. Cargar historial ──────────────────────────────────────────────────────
with open(HISTORY_PATH) as f:
    h = json.load(f)

epochs = list(range(1, len(h["accuracy"]) + 1))
print(f"Épocas entrenadas: {len(epochs)}")
print(f"Mejor val_accuracy: {max(h['val_accuracy']):.4f} (época {h['val_accuracy'].index(max(h['val_accuracy']))+1})")
print(f"Mejor val_loss:     {min(h['val_loss']):.4f} (época {h['val_loss'].index(min(h['val_loss']))+1})")

# ─── 2. Gráfica combinada: Loss + Accuracy ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Madalenos — Curvas de Entrenamiento", fontsize=14, fontweight="bold")

# Loss
ax = axes[0]
ax.plot(epochs, h["loss"],     color=COLOR_TRAIN, linewidth=2.2, label="Train Loss")
ax.plot(epochs, h["val_loss"], color=COLOR_VAL,   linewidth=2.2, label="Val Loss",   linestyle="--")
ax.axvline(x=BEST_EPOCH+1, color="green", linestyle=":", linewidth=1.5, label=f"Mejor época ({BEST_EPOCH+1})")
ax.set_title("Loss", fontsize=12)
ax.set_xlabel("Época")
ax.set_ylabel("Loss (binary crossentropy)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(epochs)

# Accuracy
ax = axes[1]
ax.plot(epochs, [v*100 for v in h["accuracy"]],     color=COLOR_TRAIN, linewidth=2.2, label="Train Accuracy")
ax.plot(epochs, [v*100 for v in h["val_accuracy"]], color=COLOR_VAL,   linewidth=2.2, label="Val Accuracy",   linestyle="--")
ax.axvline(x=BEST_EPOCH+1, color="green", linestyle=":", linewidth=1.5, label=f"Mejor época ({BEST_EPOCH+1})")
ax.set_title("Accuracy", fontsize=12)
ax.set_xlabel("Época")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 100)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(epochs)

plt.tight_layout()
out = OUT_DIR / "01_curvas_entrenamiento.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Guardado: {out.name}")

# ─── 3. Gráfica solo Accuracy (versión para portada/diapositiva) ──────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, [v*100 for v in h["accuracy"]],     color=COLOR_TRAIN, linewidth=2.5, marker="o", markersize=5, label="Train")
ax.plot(epochs, [v*100 for v in h["val_accuracy"]], color=COLOR_VAL,   linewidth=2.5, marker="s", markersize=5, label="Validación", linestyle="--")
ax.axvline(x=BEST_EPOCH+1, color="green", linestyle=":", linewidth=1.5)
ax.annotate("90.44%\n(mejor val)", xy=(BEST_EPOCH+1, 90.44),
            xytext=(BEST_EPOCH-0.5, 84), fontsize=9, color="green",
            arrowprops=dict(arrowstyle="->", color="green"))
ax.set_title("Accuracy por época — Madalenos", fontsize=13, fontweight="bold")
ax.set_xlabel("Época")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 100)
ax.set_xticks(epochs)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = OUT_DIR / "02_accuracy_destacado.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Guardado: {out.name}")

# ─── 4. Tabla-resumen de métricas ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.axis("off")
data = [
    ["Accuracy entrenamiento (época 12)", "93.07%"],
    ["Accuracy validación (mejor)       ", "90.44%"],
    ["Loss entrenamiento  (época 12)    ", "0.179 "],
    ["Loss validación (mejor)           ", "0.232 "],
    ["Épocas entrenadas                 ", "12     "],
    ["Tiempo estimado                   ", "< 10 min"],
]
headers = ["Métrica", "Valor"]
table = ax.table(cellText=data, colLabels=headers,
                 loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.4, 1.8)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#E3F2FD")
ax.set_title("Resumen de Resultados — Madalenos", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
out = OUT_DIR / "03_tabla_metricas.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Guardado: {out.name}")

# ─── 5. Resumen de arquitectura (visual) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.axis("off")
arch = [
    ("INPUT",      "128 × 128 × 3",    "#78909C"),
    ("Rescaling",  "÷ 255  →  [0,1]",  "#78909C"),
    ("Data Aug.",  "Flip · Rot · Zoom · Contrast", "#78909C"),
    ("Conv2D 32 + BN + MaxPool",  "64 × 64 × 32",   "#1565C0"),
    ("Conv2D 64 + BN + MaxPool",  "32 × 32 × 64",   "#1565C0"),
    ("Conv2D 128 + BN + MaxPool", "16 × 16 × 128",  "#1565C0"),
    ("Conv2D 256 + BN + MaxPool", "8 × 8 × 256",    "#1565C0"),
    ("Flatten",    "16.384 unidades",   "#4CAF50"),
    ("Dense 256 + BN + Dropout 50%", "256",          "#4CAF50"),
    ("Dense 128 + Dropout 30%",      "128",          "#4CAF50"),
    ("Dense 1 — sigmoid",            "0 = Chihuahua · 1 = Muffin", "#E53935"),
]
y_start = 1.0
step = 1.0 / (len(arch) + 1)
for i, (name, detail, color) in enumerate(arch):
    y = y_start - (i + 1) * step
    ax.add_patch(mpatches.FancyBboxPatch((0.05, y - step*0.35), 0.9, step*0.65,
                                         boxstyle="round,pad=0.01",
                                         linewidth=1.2, edgecolor="white",
                                         facecolor=color, alpha=0.85,
                                         transform=ax.transAxes))
    ax.text(0.5, y - step*0.02, f"{name}   —   {detail}",
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold", transform=ax.transAxes)

ax.set_title("Arquitectura CNN — Madalenos", fontsize=12, fontweight="bold")
plt.tight_layout()
out = OUT_DIR / "04_arquitectura_CNN.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Guardado: {out.name}")

# ─── 6. Comparativa baseline vs modelo final ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis("off")
comp = [
    ["Bloques Conv2D",       "3",   "4"],
    ["Filtros máximos",      "128", "256"],
    ["Data Augmentation",    "No",  "Sí (flip/rot/zoom/contrast)"],
    ["Batch Normalization",  "No",  "Sí (tras cada Conv2D y Dense)"],
    ["Dropout",              "No",  "50% y 30% en capas densas"],
    ["EarlyStopping",        "No",  "Sí (patience=5)"],
    ["ReduceLROnPlateau",    "No",  "Sí (×0.5, patience=3)"],
    ["Val Accuracy aprox.",  "~75%","90.44%"],
]
headers = ["Técnica", "Baseline", "Madalenos"]
table = ax.table(cellText=comp, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.3, 1.7)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1 and r > 0:
        cell.set_facecolor("#FFEBEE")   # rojo suave → baseline
    elif c == 2 and r > 0:
        cell.set_facecolor("#E8F5E9")   # verde suave → nuestro modelo
ax.set_title("Baseline vs Modelo Final — Madalenos", fontsize=12, fontweight="bold", pad=14)
plt.tight_layout()
out = OUT_DIR / "05_comparativa_baseline.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Guardado: {out.name}")

# ─── 7. Resumen del modelo (sin dataset) ─────────────────────────────────────
try:
    import tensorflow as tf
    import io, sys
    model = tf.keras.models.load_model(MODEL_PATH)
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    model.summary()
    sys.stdout = old_stdout
    summary_txt = buf.getvalue()
    print("✅ Modelo cargado correctamente")

    # Contar parámetros
    total_params = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"   Total parámetros: {total_params:,}")
    print(f"   Entrenables:      {trainable:,}")
except Exception as e:
    print(f"⚠️  No se pudo cargar el modelo: {e}")

print("\n" + "="*60)
print(f"Gráficas guardadas en: {OUT_DIR}")
print("="*60)
