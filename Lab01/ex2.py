import numpy as np
import matplotlib.pyplot as plt

# Ορισμός χρόνου από 0 έως 500ms (0.5s) με υψηλή ανάλυση για προσομοίωση συνεχούς σήματος
t_continuous = np.linspace(0, 0.5, 1000)

# Δημιουργία του σήματος x(t)
x_continuous = 20 * np.cos(10 * np.pi * t_continuous) - 40 * np.sin(5 * np.pi * t_continuous)

# Σχεδίαση του σήματος
plt.figure(figsize=(10, 4))
plt.plot(t_continuous * 1000, x_continuous, label='Συνεχές Σήμα x(t)') # Πολλαπλασιάζουμε με 1000 το t_continuous για να εχουμε απεικονιση σε millisecond
plt.title('Ερώτημα 2A: Αναπαράσταση Σήματος x(t)')
plt.xlabel('Χρόνος (ms)')
plt.ylabel('Πλάτος')
plt.grid(True)
plt.legend()
plt.show()