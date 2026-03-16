# 🐕 Interfaz Gradio - Chihuahua vs Muffin Classifier

Interfaz web interactiva para clasificar imágenes: ¿Es un Chihuahua o un Muffin?

## ✨ Características

- **Predicción rápida:** Sube una imagen y obtén el resultado al instante
- **Interpretabilidad:** Visualiza mapas de activación Grad-CAM para entender por qué el modelo toma sus decisiones
- **Múltiples fuentes:** Carga imágenes desde archivo, webcam, o portapapeles
- **Interfaz amigable:** Dos modos: Simple y Avanzado (con Grad-CAM)

## 🚀 Cómo ejecutar

### Opción 1: PowerShell (RECOMENDADO)

Desde la raíz del proyecto (`chihuahua_vs_muffin/`):

```powershell
.\run_gradio.ps1
```

Luego abre en navegador: **http://localhost:7860**

### Opción 2: CMD/Batch

```cmd
run_gradio.bat
```

### Opción 3: Manual (Python)

```powershell
# 1. Activar entorno virtual
.\venv\Scripts\activate

# 2. Instalar Gradio (si no lo tienes)
pip install gradio

# 3. Ejecutar la aplicación
python gradio_app.py
```

## 📱 Usar la interfaz

### Pestaña 1: Predicción Simple
1. Haz clic en el área para subir una imagen
2. Selecciona una foto de tu dispositivo, cámara web, o pégala del portapapeles
3. Haz clic en **"🔍 Predecir"**
4. Verás el resultado y un gráfico con las probabilidades

### Pestaña 2: Con Interpretabilidad (Grad-CAM)
1. Igual que arriba, pero en la pestaña "Con Interpretabilidad"
2. Además de la predicción, verás un análisis visual Grad-CAM
3. El mapa de calor muestra qué regiones influyeron en la predicción:
   - **Rojo/Amarillo** = Áreas muy importantes
   - **Azul** = Áreas menos importantes

## ⚙️ Configuración

Si necesitas cambiar el puerto (por defecto es `7860`), edita `gradio_app.py`:

```python
demo.launch(share=False, server_name="0.0.0.0", server_port=7860)  # Cambia 7860
```

## 💡 Tips

- **Webcam:** Ideal para probar en tiempo real
- **Portapapeles:** Copia una imagen y úsala directamente (Ctrl+V en el área)
- **Tamaño de imagen:** El modelo funciona con cualquier tamaño, pero fue entrenado con imágenes 128×128
- **GPU:** Si tienes GPU detectada, la interfaz la usará automáticamente

## 📊 Archivos involucrados

- `gradio_app.py` - Código principal de la interfaz
- `entrega/modelo_chihuahua_vs_muffin.keras` - Modelo entrenado
- `run_gradio.ps1` - Script PowerShell para ejecutar
- `run_gradio.bat` - Script Batch para ejecutar

## 🛑 Detener la aplicación

En la terminal donde está corriendo Gradio:
- **Ctrl + C**

## ❓ Solucionar problemas

### "ModuleNotFoundError: No module named 'gradio'"
```powershell
pip install gradio
```

### "No se encontró la carpeta del proyecto"
- Asegúrate de estar en la raíz: `C:\Users\usuario\Desktop\IABigData\repos\chihuahua_vs_muffin`
- O edita la variable `ROOT` en `gradio_app.py`

### Grad-CAM no funciona
- Es una función avanzada y a veces puede tener problemas con ciertas versiones de TF
- Usa el modo "Predicción Simple" como alternativa

---

**¡Disfruta clasificando! 🎉**
