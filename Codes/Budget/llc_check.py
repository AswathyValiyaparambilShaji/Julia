import numpy as np
import matplotlib.pyplot as plt
import os


output_dir = "/data3/aswathy/mnt/data/aswathy/MITgcm_NAS/figures/U_288x468x168.20230501T000000"
# --- Parameters ---
file_path = os.path.join(output_dir) # Example filename
shape = (1292, 1344, 90)
depth_level = 10  # Surface
dtype = ">f4"    # Big-endian float32 from your config


# 1. Read and Reshape
# We use order='F' because your extraction flattened using Fortran order
data = np.fromfile(file_path, dtype=dtype).reshape(shape, order='F')


# 2. Plot
plt.figure(figsize=(7, 4))
plt.pcolormesh(data[:, :, depth_level], cmap='RdBu_r')
plt.colorbar(label='Variable Value')
plt.title(f"Pseudocolor Plot: Level {depth_level}")
plt.xlabel("I index")
plt.ylabel("J index")
plt.tight_layout()
plt.show() 


