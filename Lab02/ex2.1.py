import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def zplane(b, a, ax=None):
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
    fig1, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax1.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
    ax1.set_title('Magnitude Response (dB)')
    ax1.set_xlabel('ω (rad/sample)'); ax1.set_ylabel(r'$20\log_{10}|H(e^{j\omega})|$ (dB)')
    fig2, ax2 = plt.subplots(figsize=(6.4, 3.6))
    ax2.plot(w, np.angle(h))
    ax2.set_title('Phase Response')
    ax2.set_xlabel('ω (rad/sample)'); ax2.set_ylabel(r'$\angle H(e^{j\omega})$ (rad)')
    return (fig1, ax1), (fig2, ax2)

# Συντελεστές του H(z) σε αύξουσες δυνάμεις του z^{-1}.
b = [0.6, 0.06, -0.12]
a = [1.0, -0.25, -0.125]
zplane(b, a)
plt.savefig('ex1_b.png', dpi=220, bbox_inches='tight')
plt.close()

# Το πλέγμα περιλαμβάνει και τα δύο άκρα και έχει ακριβώς ένα σημείο στο 0.
indices = np.arange(-128, 129)
w = indices * (np.pi / 128)
(magnitude, _), (phase, _) = plot_freqz(b, a, worN=w)
magnitude.savefig('ex1_d_magnitude.png', dpi=220, bbox_inches='tight')
phase.savefig('ex1_d_phase.png', dpi=220, bbox_inches='tight')
plt.close('all')

# Πολλαπλασιάζουμε τον παρονομαστή με (1 - z^{-1}).
# Στο ω=0 υπάρχει πόλος. Αφήνουμε κενό ανάμεσα στους δύο κλάδους.
a_with_pole = np.convolve(a, [1.0, -1.0])
(magnitude, mag_ax), (phase, phase_ax) = plot_freqz(
    b, a_with_pole, worN=w[indices != 0]
)
for fig, ax, title, filename in [
    (magnitude, mag_ax, 'Magnitude response with a pole at z = 1',
     'ex1_e_magnitude.png'),
    (phase, phase_ax, 'Phase response with a pole at z = 1',
     'ex1_e_phase.png'),
]:
    line = ax.lines[0]
    line.set_data(w, np.insert(line.get_ydata(), 128, np.nan))
    ax.axvline(0, color='0.45', linestyle='--', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlim(-np.pi, np.pi)
    fig.savefig(filename, dpi=220, bbox_inches='tight')
plt.close('all')
