import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parametere
V1 = V2 = V3 = 5  # Volum [L]
q = 1  # Strømningshastighet [L/min]
cAf = 2  # Inngangskonsentrasjon [mol/L]

def diff(t, x):
    cA1, cA2, cA3 = x
    dcA1dt = q / V1 * (cAf - cA1)
    dcA2dt = q / V2 * (cA1 - cA2)
    dcA3dt = q / V3 * (cA2 - cA3)
    return [dcA1dt, dcA2dt, dcA3dt]

def euler_explisitt(cA0, t_eval, h):
    n = len(t_eval)
    cA = np.zeros((n, 3))
    cA[0] = cA0
    
    for i in range(1, n):
        cA1, cA2, cA3 = cA[i-1]
        cA[i, 0] = cA1 + h * (q / V1 * (cAf - cA1))
        cA[i, 1] = cA2 + h * (q / V2 * (cA1 - cA2))
        cA[i, 2] = cA3 + h * (q / V3 * (cA2 - cA3))
    
    return cA

def euler_implisitt(cA0, t_eval, h):
    n = len(t_eval)
    cA = np.zeros((n, 3))
    cA[0] = cA0
    
    for i in range(1, n):
        cA1_old, cA2_old, cA3_old = cA[i-1]
        cA[i, 0] = (cA1_old + h * (q / V1 * cAf)) / (1 + h * q / V1)
        cA[i, 1] = (cA2_old + h * (q / V2 * cA[i, 0])) / (1 + h * q / V2)
        cA[i, 2] = (cA3_old + h * (q / V3 * cA[i, 1])) / (1 + h * q / V3)
    
    return cA

# Initialverdier og tidssteg
cA0 = [1, 1, 1]
t_eval = np.linspace(0, 50, 100)
h = t_eval[1] - t_eval[0]

# Løsning med eksplisitt og implisitt Euler
cA_explisitt = euler_explisitt(cA0, t_eval, h)
cA_implisitt = euler_implisitt(cA0, t_eval, h)

# Løsning med solve_ivp
sol = solve_ivp(diff, (0, 50), cA0, t_eval=t_eval)

# Plot resultater
plt.plot(t_eval, cA_explisitt[:, 0], label="Eksplisitt Euler (Tank 1)", linestyle='--')
plt.plot(t_eval, cA_explisitt[:, 1], label="Eksplisitt Euler (Tank 2)", linestyle='--')
plt.plot(t_eval, cA_explisitt[:, 2], label="Eksplisitt Euler (Tank 3)", linestyle='--')

plt.plot(t_eval, cA_implisitt[:, 0], label="Implisitt Euler (Tank 1)", linestyle='-')
plt.plot(t_eval, cA_implisitt[:, 1], label="Implisitt Euler (Tank 2)", linestyle='-')
plt.plot(t_eval, cA_implisitt[:, 2], label="Implisitt Euler (Tank 3)", linestyle='-')

plt.plot(t_eval, sol.y[0], label="solve_ivp (Tank 1)", linestyle=':')
plt.plot(t_eval, sol.y[1], label="solve_ivp (Tank 2)", linestyle=':')
plt.plot(t_eval, sol.y[2], label="solve_ivp (Tank 3)", linestyle=':')

plt.xlabel("Tid [min]")
plt.ylabel("Konsentrasjon [mol/l]")
plt.title("Konsentrasjon av A over tid (Eksplisitt vs Implisitt Euler vs solve_ivp)")
plt.legend()
plt.grid(True)
plt.show()

# Beregn tiden for hver tank der konsentrasjonen overskrider 1.8 mol/l
def finn_tid(cA_arr, threshold=1.8):
    for i, val in enumerate(cA_arr):
        if val > threshold:
            return i
    return -1  # Hvis den aldri overskrider threshold

# Finn tid for eksplisitt Euler
for i, tank in enumerate(["Tank 1", "Tank 2", "Tank 3"]):
    tid_explisitt = finn_tid(cA_explisitt[:, i])
    if tid_explisitt != -1:
        print(f"Eksplisitt Euler: Det tar {t_eval[tid_explisitt]:.2f} min for {tank} å overskride 1.8 mol/l")

# Finn tid for implisitt Euler
for i, tank in enumerate(["Tank 1", "Tank 2", "Tank 3"]):
    tid_implisitt = finn_tid(cA_implisitt[:, i])
    if tid_implisitt != -1:
        print(f"Implisitt Euler: Det tar {t_eval[tid_implisitt]:.2f} min for {tank} å overskride 1.8 mol/l")

# Finn tid for solve_ivp
for i, tank in enumerate(["Tank 1", "Tank 2", "Tank 3"]):
    tid_ivp = finn_tid(sol.y[i])
    if tid_ivp != -1:
        print(f"solve_ivp: Det tar {t_eval[tid_ivp]:.2f} min for {tank} å overskride 1.8 mol/l")

