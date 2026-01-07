import pickle
import matplotlib.pyplot as plt

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
HISTORY_FILE = 'history_tbnet_comp_40ep.pkl'
EPOCHS = 40

# ---------------------------
# 1. Cargar historial
# ---------------------------
with open(HISTORY_FILE, 'rb') as f:
    history = pickle.load(f)

train_acc = history['accuracy']
val_acc   = history['val_accuracy']

# ---------------------------
# 2. Estilo EXACTO del ejemplo
# ---------------------------
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# ---------------------------
# 3. Crear figura MÁS ALTA y MÁS DELGADA
# ---------------------------
plt.figure(figsize=(5, 7), dpi=300)  #  <-- AQUÍ SE AJUSTA LA FORMA (largo, delgado)

# Fondo gris
ax = plt.gca()
ax.set_facecolor("#d3d3d3")
for spine in ax.spines.values():
    spine.set_color('black')

# ---------------------------
# 4. Plot líneas EXACTAS
# ---------------------------
plt.plot(train_acc, color="green", linewidth=2.5, label="Entrenamiento")
plt.plot(val_acc,   color="purple", linewidth=2.5, label="Validación")

# ---------------------------
# 5. Ejes y estética
# ---------------------------
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.xticks(range(0, EPOCHS + 1, 5))
plt.ylim(0.68, 1.00)

# ---------------------------
# 6. Leyenda abajo derecha
# ---------------------------
plt.legend(
    facecolor="white",
    framealpha=1.0,
    edgecolor="black",
    loc='lower right'
)

# ---------------------------
# 7. Resultado final
# ---------------------------
plt.tight_layout()
plt.savefig("accuracy_vertical.png", dpi=300, bbox_inches='tight')
plt.show()
