import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Α)
# Ορισμός πολυωνύμων αριθμητή και παρονομαστή (σε δυνάμεις του z^-1). Διαιρούμε με το 4 ώστε ο 
# πρώτος όρος να είναι κανονικοιποιημένος επειδή έτσι απαιτεί ο αλγόρυθμος της διαίρεσης
b = [2.0, -0.6]
a = [1.0, -1.25, 0.07, 0.2205]

# Υπολογισμός ανάλυσης σε απλά κλάσματα (residues, poles, direct terms)
r, p, k = signal.residuez(b, a)

print("Υπόλοιπα (Residues r):", r)
print("Πόλοι (Poles p):", p)
print("Άμεσοι όροι (Direct terms k):", k)

# Β)
# Δημιουργία σήματος κρούσης δ[n] μήκους N = 15
n_samples = 15
delta = signal.unit_impulse(n_samples)

# Υπολογισμός κρουστικής απόκρισης h[n]
h_n = signal.lfilter(b, a, delta)

print("Υπολογισμένη κρουστική απόκριση h[n]:")
for i, val in enumerate(h_n):
    print("h[{0}] = {1}".format(i, val))

def zplane(b, a, ax=None):
    """Plot zeros and poles of H(z)=B(z)/A(z)."""
    b = np.atleast_1d(b); a = np.atleast_1d(a)
    if a[0] != 1:
        b = b / a[0]; a = a / a[0]
    z = np.roots(b) if len(b) > 1 else np.array([])
    p = np.roots(a) if len(a) > 1 else np.array([])
    if ax is None:
        fig, ax = plt.subplots()
    uc = plt.Circle((0,0), 1, fill=False, linestyle='--')
    ax.add_artist(uc)
    ax.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
    ax.scatter(np.real(p), np.imag(p), marker='x', label='Poles')
    ax.set_xlabel('Re{z}'); ax.set_ylabel('Im{z}')
    ax.axhline(0, linewidth=0.5); ax.axvline(0, linewidth=0.5)
    r = 1.3*max(1, np.max(np.abs(np.concatenate([z, p])))) if z.size or p.size else 1.2
    ax.set_xlim([-r, r]); ax.set_ylim([-r, r])
    ax.set_aspect('equal', 'box'); ax.legend(); ax.set_title('Pole-Zero Map')
    return z, p

def plot_freqz(b, a, worN=1024):
    w, h = signal.freqz(b, a, worN=worN)
    fig1, ax1 = plt.subplots()
    ax1.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
    ax1.set_title('Magnitude Response (dB)')
    ax1.set_xlabel('ω (rad/sample)'); ax1.set_ylabel('|H(e^{jω})| (dB)')
    fig2, ax2 = plt.subplots()
    ax2.plot(w, np.unwrap(np.angle(h)))
    ax2.set_title('Phase Response')
    ax2.set_xlabel('ω (rad/sample)'); ax2.set_ylabel('∠H(e^{jω}) (rad)')
    return (fig1, ax1), (fig2, ax2)

zplane(b, a)
plot_freqz(b, a)
plt.show()