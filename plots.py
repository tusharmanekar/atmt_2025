import matplotlib.pyplot as plt

# ----- ASSIGNMENT 1 -----
epochs_a1 = [0, 1, 2, 3, 4, 5, 6]
train_a1 = [1.952, 1.576, 1.47, 1.413, 1.376, 1.35, 1.33]
val_a1   = [2.33, 2.05, 1.94, 1.85, 1.79, 1.77, 1.75]

# ----- ASSIGNMENT 3 -----
epochs_a3 = [0, 1, 2, 3, 4, 5]
train_a3 = [1.889, 1.54, 1.441, 1.387, 1.352, 1.327]
val_a3   = [2.00, 1.79, 1.69, 1.63, 1.61, 1.58]

# --------------------------
# Plot 1: TRAIN LOSS
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs_a1, train_a1, marker='o', label="Assignment 1 — Train Loss")
plt.plot(epochs_a3, train_a3, marker='o', label="Assignment 3 — Train Loss")

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# Plot 2: VAL LOSS
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs_a1, val_a1, marker='o', label="Assignment 1 — Val Loss")
plt.plot(epochs_a3, val_a3, marker='o', label="Assignment 3 — Val Loss")

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
