import rpy2.robjects as robjects
import numpy as np
import matplotlib.pyplot as plt

legend_font = 10
labels_font = 13


path = './data/'
path_nudging = './results_nudging/'

# load results for R-learning 
robjects.r['load'](f"{path}/rhos_rlearning2.RData")
rhos_rl2 = np.array(robjects.r['rhos'])
robjects.r['load'](f"{path}/difVs_rlearning2.RData")
vs_rl2 = np.array(robjects.r['errdos'])

# load results for SSQP
robjects.r['load'](f"{path}/rhos_sspq.RData")
rhos_sspq = np.array(robjects.r['rhos'])
robjects.r['load'](f"{path}/difVs_sspq.RData")
vs_sspq = np.array(robjects.r['errdos'])

# load results for R-SMART
robjects.r['load'](f"{path}/rhos_rsmart2.RData")
rhos_rsmart = np.array(robjects.r['rhos'])
robjects.r['load'](f"{path}/difVs_rsmart.RData")
vs_rsmart = np.array(robjects.r['errdos'])


# load results for  SMART
robjects.r['load'](f"{path}/rhos_smart.RData")
rhos_smart = np.array(robjects.r['rhos'])
robjects.r['load'](f"{path}/difVs_smart.RData")
vs_smart = np.array(robjects.r['errdos'])


# load results for nudging
rhos_opt = np.load(f'{path_nudging}/rhos.npy')
vs_opt = np.load(f'{path_nudging}/values_sI.npy')


fig, (ax1, ax2) = plt.subplots(1, 2)

rho_base = 3.28
ax1.plot(np.abs(rho_base-rhos_sspq), label='SSPQ', color='tab:brown')
ax1.plot(np.abs(rho_base-rhos_rsmart), label='R-SMART', color='tab:green')
ax1.plot(np.abs(rho_base-rhos_smart), label='SMART',  color='tab:orange')
ax1.plot(np.abs(rho_base-rhos_rl2), label='R Learning', color='tab:red')
ax1.plot(np.abs(rho_base-rhos_opt), label='Optimal nudging', color='tab:blue')
ax1.set_yscale('log')
# ax1.set_xticklabels([0,0,1,2,3,4,5])
ax1.set_xlabel(f'x10^6 steps')
ax1.set_ylabel(f'|ρ*-ρt|')
ax1.legend()


ax2.plot(vs_sspq, label='SSPQ', color='tab:brown')
ax2.plot(vs_rsmart, label='R-SMART', color='tab:green')
ax2.plot(vs_smart, label='SMART', color='tab:orange')
ax2.plot(vs_rl2, label='R Learning', color='tab:red')
ax2.plot(vs_opt, label='Optimal nudging', color='tab:blue')
ax2.set_yscale('log')
# ax2.set_xticklabels([0,0,1,2,3,4,5])
ax2.set_xlabel(f'x10^6 steps')
ax2.set_ylabel(r'||V*-Vt||₂')
ax2.legend()


plt.show()