import numpy as np
import matplotlib.pyplot as plt

# Ορισμός x[n] από -4 έως 4
n_x = np.arange(-4, 5)
x = 10 - n_x

# Ορισμός y[n] από -2 έως 5
n_y = np.arange(-2, 6)
y = 0.8**(n_y + 1)

# Δημιουργία γραφικών παραστάσεων για τα αρχικά σήματα
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(n_x, x)
plt.title('Σήμα διακριτού χρόνου x[n]')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(n_y, y)
plt.stem(n_y, y)
plt.title('Σήμα διακριτού χρόνου y[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)

# Υπολογισμός διαστήματος ορισμού του αποτελέσματος της συνέλιξης.
# Αν το x[n] έχει μήκος L1 και ξεκινά στο n1, και το y[n] έχει μήκος L2 και ξεκινά στο n2,
# τότε η συνέλιξή τους ξεκινά στο n1 + n2 και έχει μήκος L1 + L2 - 1.
start_conv = n_x[0] + n_y[0]
len_conv = len(x) + len(y) - 1
n_conv = np.arange(start_conv, start_conv +len_conv)

# Υπολογισμός συνέλιξης
conv_result = np.convolve(x, y)

# Δημιουργία γραφικών παραστάσεων συνέλιξης
plt.subplot(3, 1, 3)
plt.stem(n_conv, conv_result)
plt.title('Αποτέλεσμα Συνέλιξης x[n] * y[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()


