import numpy as np
import matplotlib.pyplot as plt

# Ορισμός χρόνου από 0 έως 500ms (0.5s) με υψηλή ανάλυση για προσομοίωση συνεχούς σήματος
t_continuous = np.linspace(0, 0.5, 1000)

# Δημιουργία του σήματος x(t)
x_continuous = 20 * np.cos(10 * np.pi * t_continuous) - 40 * np.sin(5 * np.pi * t_continuous)

# Σχεδίαση του σήματος
plt.figure(figsize=(10, 4))
plt.plot(t_continuous * 1000, x_continuous, label='Συνεχές Σήμα x(t)') # Multilpy by 1000 t_continuous to convert to milliseconds
plt.title('Ερώτημα 2A: Αναπαράσταση Σήματος x(t)')
plt.xlabel('Χρόνος (ms)')
plt.ylabel('Πλάτος')
plt.grid(True)
plt.legend()
plt.show()

Fs = 8000.0 # Συχνότητα δειγματοληψίας (Hz)
DT = 1.0 / Fs # Περίοδος δειγματοληψίας (s)
t = np.arange(-500, 500, DT) # Χρόνος: [-10, 10)
# Σήμα: άθροισμα 5 Hz και 2.5 Hz ημιτόνων
x = 20 * np.sin(10 * np.pi * t) - 40 * np.sin( 5 * np.pi * t)
# NFFT: μεγαλύτερη δύναμη του 2
NFFT = 1 << int(np.ceil(np.log2(len(x))))
# FFT με κεντράρισμα (DT για προσέγγιση ολοκλ.)
X = np.fft.fftshift(np.fft.fft(x, NFFT) * DT / 1000) # Divide by 1000 to account for s -> ms 
# Διάνυσμα συχνότητας (Hz), κεντραρισμένο
F = np.fft.fftshift(np.fft.fftfreq(NFFT, d=DT))
# Φάσμα πλάτους
plt.figure()
plt.plot(F, np.abs(X))
plt.xlim([-15, 15])
plt.title('Amplitude Spectrum of x(t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X(f)|')
plt.grid(True, alpha=0.3)
plt.show()

# Επιθυμητές περίοδοι δειγματοληψίας
Ts_values = [1/20, 1/10, 1/5]
t_continuous = np.linspace(0, 0.5, 1000)


for Ts in Ts_values:
    Fs = 1/Ts
    n_start = int(np.ceil(0 * Fs))
    n_end = int(np.floor(0.5 * Fs))
    n = np.arange(n_start, n_end + 1)
    t_d = n * Ts
    
    # Υπολογισμός διακριτού σήματος
    x_discrete = 20 * np.cos(10 * np.pi * t_d) - 40 * np.sin(5 * np.pi * t_d)
    
    # Σχεδίαση του συνεχούς και του διακριτού σήματος στο ίδιο γράφημα
    plt.figure(figsize=(10, 4))
    plt.plot(t_continuous, x_continuous, label='Συνεχές x(t)', color='blue')
    plt.stem(t_d, x_discrete, label=f'Διακριτό (Ts={Ts}s)', linefmt='-r')
    plt.title(f'Ερώτημα 2Γ: Δειγματοληψία με περίοδο Ts = {Ts} s')
    plt.xlabel('Χρόνος (s)')
    plt.ylabel('Πλάτος')
    plt.grid(True)
    plt.legend()
    plt.show()
    