import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Κανονικοποιημένοι συντελεστές του H(z) σε δυνάμεις του z^{-1}.
b = [2.0, -0.6]
a = [1.0, -1.25, 0.07, 0.2205]

# Αναλύουμε το H(z) σε απλά κλάσματα για να εντοπίσουμε τους πόλους και τα υπόλοιπα.
residues, poles, direct = signal.residuez(b, a)
print("Residues:", residues)
print("Poles:", poles)
print("Direct terms:", direct)

# Η είσοδος δ[n] δίνει την κρουστική απόκριση για μηδενικές αρχικές συνθήκες.
n = np.arange(15)
h_filter = signal.lfilter(b, a, signal.unit_impulse(n.size))

# Η αναλυτική μορφή προκύπτει από τα εκθετικά των πόλων και τα αντίστοιχα υπόλοιπα.
h_formula = -(8 / 3) * 0.7**n + (26 / 75) * (-0.35)**n + (108 / 25) * 0.9**n

# Υπολογίζουμε τη μεγαλύτερη απόλυτη διαφορά μεταξύ των δύο αποκρίσεων.
error = np.max(np.abs(h_filter - h_formula))

# Η δοκιμή αποτυγχάνει αν κάποια τιμή αποκλίνει περισσότερο από την ανοχή 1e-12.
np.testing.assert_allclose(h_filter, h_formula, rtol=0, atol=1e-12)

# Εκτυπώνουμε τις τιμές δίπλα-δίπλα για άμεση σύγκριση ανά δείγμα.
print("n      h[n] (formula)    h[n] (lfilter)")
for i, analytical, numerical in zip(n, h_formula, h_filter):
    print(f"{i:2d}     {analytical:12.8f}     {numerical:12.8f}")
print(f"Maximum absolute error: {error:.3e}")

# Συγκρίνουμε τα αριθμητικά δείγματα με τις αναλυτικές τιμές στο ίδιο γράφημα.
fig, ax = plt.subplots(figsize=(7, 2.5), layout='constrained')
ax.stem(n, h_filter, label='lfilter', basefmt=' ')
ax.plot(n, h_formula, 'o', markerfacecolor='none', markeredgecolor='tab:orange',
        markersize=8, label='Analytical expression')
ax.set(title='Impulse response', xlabel='n (samples)', ylabel='h[n]')
ax.set_xticks(n)
ax.legend()
fig.savefig('ex2_impulse.png', dpi=220, bbox_inches='tight')
plt.close(fig)
