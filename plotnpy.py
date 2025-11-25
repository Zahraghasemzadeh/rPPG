import numpy as np
import matplotlib.pyplot as plt

# --- Load the .npy file ---
rppg_data = np.load("outputs//rppg_20251108-114228.npy", allow_pickle=True)
print(rppg_data.shape)



# --- Plot each RPPG signal ---
plt.figure(figsize=(10, 6))

plt.plot(rppg_data)  # since you extracted rows 25–34

plt.title("RPPG Signals")
plt.xlabel("Sample Index")
plt.ylabel("RPPG Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


