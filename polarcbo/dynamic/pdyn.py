#%%
class ParticleDynamic(object):
    def __init__(self, x, V, beta=1.0):
        self.x = x.copy()
        self.num_particles = x.shape[0]
        self.V = V
        self.energy = None