import numpy as np

atoms_to_save = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 18])

u_cdl = np.load("u_cdl.npy")
v_cdl = np.load("v_cdl.npy")

np.save("u_cdl_modified.npy", u_cdl[atoms_to_save])
np.save("v_cdl_modified.npy", v_cdl[atoms_to_save])
