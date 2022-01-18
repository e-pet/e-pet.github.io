from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

# Example 1: Impulse
x1 = np.arange(0, 7)
x1_dense = np.linspace(start=0, stop=6, num=1000)
y1 = np.array([0, 0, 0, 1, 0, 0, 0])
f_cubic_splines = interpolate.interp1d(x1, y1, kind='cubic', fill_value='extrapolate')
f_akima = interpolate.Akima1DInterpolator(x1, y1)
f_pchip = interpolate.PchipInterpolator(x1, y1)

plt.figure(figsize=(6, 4))
plt.plot(x1, y1, label='Data', linestyle='None', marker='x')
plt.plot(x1_dense, f_cubic_splines(x1_dense), label='Cubic Splines')
plt.plot(x1_dense, f_akima(x1_dense), label='Akima')
plt.plot(x1_dense, f_pchip(x1_dense), label='Pchip')
plt.legend()
plt.savefig('interpolation-impulse.png', bbox_inches='tight')

# Example 2: Sampled Sine
x2 = np.linspace(start=0, stop=5.5, num=4)
x2_dense = np.linspace(start=0, stop=5.5, num=1000)
y2 = np.sin(x2)
f_cubic_splines = interpolate.interp1d(x2, y2, kind='cubic', fill_value='extrapolate')
f_akima = interpolate.Akima1DInterpolator(x2, y2)
f_pchip = interpolate.PchipInterpolator(x2, y2)

plt.figure(figsize=(6, 4))
plt.plot(x2, y2, label='Data', linestyle='None', marker='x')
plt.plot(x2_dense, f_cubic_splines(x2_dense), label='Cubic Splines')
plt.plot(x2_dense, f_akima(x2_dense), label='Akima')
plt.plot(x2_dense, f_pchip(x2_dense), label='Pchip')
plt.legend()
plt.savefig('interpolation-sine.png', bbox_inches='tight')

plt.show()
