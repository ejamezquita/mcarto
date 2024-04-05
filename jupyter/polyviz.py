import polyscope as ps
import numpy as np

ps.init()

# define the resolution and bounds of the grid
dims = (20, 20, 20)
bound_low = (-3., -3., -3.)
bound_high = (3., 3., 3.)

# register the grid
ps_grid = ps.register_volume_grid("sample grid", dims, bound_low, bound_high)

# your dimX*dimY*dimZ buffer of data
scalar_vals = np.zeros(dims)

# add a scalar function on the grid
ps_grid.add_scalar_quantity("node scalar1", scalar_vals, 
                            defined_on='nodes', vminmax=(-5., 5.), enabled=True)

ps.show()
