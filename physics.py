# Final project in SI1336

import math

import numpy as np

a = 1e-3
G = 1e-11*6.67430  # N⋅m2/kg2

class Particle:
    def __init__(self,q0x,q0y,p0x,p0y, m):
        self.q0x = q0x
        self.q0y = q0y
        self.p0x = p0x
        self.p0y = p0y
        self.qx = q0x
        self.qy = q0y
        self.px = p0x
        self.py = p0y
        self.m = m

    def copy(self):
        return Particle(self.qx, self.qy, self.px, self.py, self.m)

class InitialState:
    def __init__(self, initialparticles):
        self.initialparticles = initialparticles

    def get_state(self):
        particles = []
        for particle in self.initialparticles:
            particles.append(particle.copy())
        return particles

class Integrator:
    def __init__(self, particles, dt):
        self.particles = particles
        self.N = len(particles)
        self.dt = dt
    
    def integrate(self):
        # Should implement this method
        pass

class SymplecticEuler(Integrator):
    def dHdq(self, i):
        dHdq_x = 0
        dHdq_y = 0
        for i in range(self.N):
            for j in range(i+1, self.N):
                denom = ((self.particles[i].qx-self.particles[j].qx)**2 + (self.particles[i].qy-self.particles[j].qy)**2)**(3/2)
                dHdq_x += G*self.particles[i].m*self.particles[j].m*(self.particles[i].qx - self.particles[j].qx)/denom
                dHdq_y += G*self.particles[i].m*self.particles[j].m*(self.particles[i].qy - self.particles[j].qy)/denom
        return dHdq_x, dHdq_y

    def dHdp(self, i):
        return (self.particles[i].px/self.particles[i].m, self.particles[i].py/self.particles[i].m)

    def integrate(self):
        for i,particle in enumerate(self.particles):
            dHdq_x, dHdq_y = self.dHdq(i)
            dHdp_x, dHdp_y = self.dHdp(i)
            particle.px = particle.px - self.dt*dHdq_x
            particle.py = particle.py - self.dt*dHdq_y
            particle.qx = particle.qx + self.dt*dHdp_x
            particle.qx = particle.qx + self.dt*dHdp_y

class System:
    def __init__(self, particles, integrator, tmax, dt):
        self.particles = particles
        self.particle_positions = np.array([]) # todo define shape 
        self.integrator = integrator(particles, dt)
        self.tmax = tmax
        self.dt = dt
        self.t = 0

    def simulate(self):
        while self.t < self.tmax:
            self.integrator.integrate()
            self.t += self.dt
            print(self.particles[0].qx)

    def store_particle_positions(self):
        pass


if __name__ == "__main__":
    a=0.5
    particle1 = Particle(1-a, 0,0,math.sqrt((1+a)/(1-a)), 1)
    particle2 = Particle(0,0,0,0, 1)
    particles = [particle1, particle2]
    initial_state = InitialState(particles)
    system = System(initial_state.get_state(), SymplecticEuler, 1, 0.01)
    system.simulate()