import CMap2D
import numpy as np

cmap2d = CMap2D.CMap2D(".", "office_full")
ij = cmap2d.xy_to_ij(np.array([[0, 0], [1,1]], dtype=np.float32))
print(ij)
xy = cmap2d.ij_to_xy(ij.astype(np.float32))
print(xy)
