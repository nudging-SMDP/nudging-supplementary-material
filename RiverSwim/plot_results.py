import numpy as np
import matplotlib.pyplot as plt


path_nudging = './results_nudging/'
path_rlearning = './results_rlearning/'


rhos_q = np.load(path_nudging + 'rhos.npy')
rhos_r = np.load(path_rlearning + 'rhos.npy')


fig, (ax1) = plt.subplots(1, 1)


steps = np.arange(0,len(rhos_q))/1000
rho_base = 0.4286224337994642
ax1.plot(steps, np.abs(rhos_q-rho_base), label='Optimal nudging', color = 'tab:blue')
steps = np.arange(0,len(rhos_r))/1000
ax1.plot(steps, np.abs(rhos_r-rho_base), label='R Learning', color='tab:red')
ax1.set_yscale('log')
ax1.legend()
ax1.set_xlabel(f'x10^6 steps')
ax1.set_ylabel(f'|ρ*-ρt|')


plt.show()

