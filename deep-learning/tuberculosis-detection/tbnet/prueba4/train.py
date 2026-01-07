import pickle
import matplotlib.pyplot as plt

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
HISTORY_FILE = 'history_tbnet_40ep.pkl'
EPOCHS = 40   # Número de épocas del entrenamiento

# ---------------------------
# 1. Cargar Historial
# ---------------------------
with open(HISTORY_FILE, 'rb') as f:
    history = pickle.load(f)

train_loss = history['loss']
val_loss   = history['val_loss']

# ---------------------------
# 2. Configurar estilo grande (similar al gráfico de la izquierda)
# ---------------------------
plt.rcParams['font.size'] = 20        # tamaño general de letra
plt.rcParams['axes.titlesize'] = 22   # título
plt.rcParams['axes.labelsize'] = 22   # etiquetas X/Y
plt.rcParams['xtick.labelsize'] = 18  # números eje X
plt.rcParams['ytick.labelsize'] = 18  # números eje Y
plt.rcParams['legend.fontsize'] = 20  # leyenda

# ---------------------------
# 3. Crear figura grande
# ---------------------------
plt.figure(figsize=(7, 7), dpi=300)   # cuadrada, grande y en alta resolución

# ---------------------------
# 4. Graficar curvas
# ---------------------------
plt.plot(train_loss, label='Entrenamiento', linewidth=3)
plt.plot(val_loss,   label='Validación',   linewidth=3)

# ---------------------------
# 5. Ejes y estilos
# ---------------------------
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el Entrenamiento')

plt.xticks(range(0, EPOCHS + 1, 5))   # 0, 5, 10, ..., 40
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

# ---------------------------
# 6. Guardar y mostrar
# ---------------------------
plt.savefig('loss_densenet_grande.png', dpi=300, bbox_inches='tight')
plt.show()
