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

# plot_freqz αλλαγμένη ώστε να ώστε να έχει τα label του δ)
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

# plot_freqz αλλαγμένη ώστε να ώστε να έχει τα label του ε)
def zplot_freqz(b, a, worN=1024):
    w, h = signal.freqz(b, a, worN=worN)
    fig1, ax1 = plt.subplots()
    ax1.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
    ax1.set_title('Magnitude Response με πόλο στο z=1 (dB)')
    ax1.set_xlabel('ω (rad/sample)'); ax1.set_ylabel('|H(e^{jω})| (dB)')
    fig2, ax2 = plt.subplots()
    ax2.plot(w, np.unwrap(np.angle(h)))
    ax2.set_title('Phase Response με πόλο στο z=1')
    ax2.set_xlabel('ω (rad/sample)'); ax2.set_ylabel('∠H(e^{jω}) (rad)')
    return (fig1, ax1), (fig2, ax2)

# β)
# Συντελεστές H(z) που υπολογίστηκαν
b = [0.6, 0.06, -0.12]
a = [1.0, -0.25, -0.125]

# Σχεδίαση διαγράμματος
zplane(b, a)
plt.savefig('Lab02/ex1_b.png')

# δ)
# Δημιουργία διαστήματος [-π, π] με βήμα π/128
w_vec = np.arange(-np.pi, np.pi, np.pi/128)

# Σχεδίαση απόκρισης συχνότητας
(fig_mag_d, _), (fig_phase_d, _) = plot_freqz(b, a, worN=w_vec)
fig_mag_d.savefig('Lab02/ex1_d_magnitude.png')
fig_phase_d.savefig('Lab02/ex1_d_phase.png')

# ε)
# Πρόσθεση επιπλέον πόλου στο z=1
# Αυτό σημαίνει παράγοντας 1-z^-1 άρα [1, -1]
pole_at_one = [1.0, -1.0]
a_new = np.convolve(a, pole_at_one)

# Δημιουργία διαστήματος [-π, π]και σχεδίαση
w_vec = np.arange(-np.pi, np.pi, np.pi/128)
# Σχεδίαση με πόλο στο z=1
(fig_mag_e, _), (fig_phase_e, _) = zplot_freqz(b, a_new, worN=w_vec)
fig_mag_e.savefig('Lab02/ex1_e_magnitude.png')
fig_phase_e.savefig('Lab02/ex1_e_phase.png')



