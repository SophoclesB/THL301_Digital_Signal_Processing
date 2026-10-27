import numpy as np
import matplotlib.pyplot as plt

# A) 
# Ορίζουμε πλήθος δειγμάτων και συχνότητα
N = 128
fs_a = 400.0 # To 400 είναι αρκετά μεγαλύτερο από το 2fmax = 120 ώστε να μην
# αλλοιωθεί από το φενόμενο της απικάληψης. Θα δούλευε οποιαδήπτε άλλη τιμη μεγαλύτερη του 400
# αλλά αυτό μας δίνει μια ασφαλή απόσταση

T_a = 1.0 / fs_a
n = np.arange(N)
t = n * T_a

# Ορισμός σήματος x(t)
f1 = 20.0
f2 = 60.0
phi = np.deg2rad(30)
x_t = 12 * np.cos(2 * np.pi * f1 * t) - 18 * np.sin(2 * np.pi * f2 * t + phi)

# Υπολογισμός FFT 
X_fft = np.fft.fft(x_t)
freqs_a = np.fft.fftfreq(N, d = T_a)

# Σχεδίαση φάσματος
plt.figure(figsize=(10, 4))
plt.stem(freqs_a[:N//2], np.abs(X_fft[:N//2]) / N, basefmt=" ")
plt.title('Φάσμα ΠΛάτους Σήματος x(t)')
plt.xlabel('Συχνότητα (Hz)')
plt.ylabel('|X(f)|')
plt.grid(True)
plt.savefig('LabCode/Lab01/ex3_a.png')

# Β)
# Ορίζουμε πλήθος δειγμάτων και συχνότητα
fs_b = 20000.0
N_b = 1000
n_b = np.arange(N_b)
t_b = n_b / fs_b

# Συναρτήσεις για τις χαμηλές συχνότητες
f_low = [500, 1000, 1500, 2000]

plt.figure(figsize=(12, 6))
i = 1
for f0 in f_low:
    # Δημιουργία του σήματος
    x_b = np.sin(2 * np.pi * f0 * t_b)
    # Υπολογισμός FFT
    X_f = np.fft.fft(x_b)
    freqs_b = np.fft.fftfreq(N_b, d = 1 / fs_b)
    # Σχεδίαση στο ανάλογο subplot
    plt.subplot(2, 2, i)
    plt.plot(freqs_b[:N_b//2], np.abs(X_f[:N_b//2]))
    plt.title(f'f0 = {f0}Hz (Χωρίς Aliasing)')
    plt.xlabel('Hz')
    plt.grid(True)
    i +=1

plt.tight_layout()
plt.savefig('LabCode/Lab01/ex3_b.png')

# Ορισμός συχνοτήτων μεταξύ 18000 και 19500 με βήμα 500
f_high = [18000, 18500, 19000, 19500]

plt.figure(figsize=(12, 6))
i = 1
for f0 in f_high:
    # Δημιουργία σήματος
    x_b = np.sin(2 * np.pi * f0 * t_b)
    # Υπολογισμός FFT
    X_f = np.fft.fft(x_b)
    freqs_b = np.fft.fftfreq(N_b, d = 1 / fs_b)
    # Σχεδίαση στο ανάλογο subplot
    plt.subplot(2, 2, i)
    plt.plot(freqs_b[:N_b//2], np.abs(X_f[:N_b//2]), color='orange')
    plt.title(f'f0 = {f0}Hz (με Aliasing)')
    plt.xlabel('Hz')
    plt.grid(True)
    i += 1

plt.tight_layout()
plt.savefig('LabCode/Lab01/ex3_c.png')
