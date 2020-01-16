# Final project in SI1336

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


G = 6.67408*1e-11  # N⋅m2/kg2

def norm(a,b):
    return math.hypot(a, b)

class Particle:
    def __init__(self,q0x,q0y,p0x,p0y, m, fixed=False):
        self.q0x = q0x
        self.q0y = q0y
        self.p0x = p0x
        self.p0y = p0y
        self.qx = q0x
        self.qy = q0y
        self.px = p0x
        self.py = p0y
        self.m = m
        self.fixed = fixed

    def kinetic_energy(self):
        return norm(self.px, self.py)**2/self.m/2

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
                denom = ((self.particles[i].qx - self.particles[j].qx)**2 + (self.particles[i].qy - self.particles[j].qy)**2)**(3/2)
                dHdq_x += G*self.particles[i].m*self.particles[j].m*(self.particles[i].qx - self.particles[j].qx)/denom
                dHdq_y += G*self.particles[i].m*self.particles[j].m*(self.particles[i].qy - self.particles[j].qy)/denom
        return dHdq_x, dHdq_y

    def dHdp(self, i):
        return (self.particles[i].px/self.particles[i].m, self.particles[i].py/self.particles[i].m)

    def integrate(self):
        # calculate the gradients first, then loop again to update
        # this is to avoid particle's being updated based on the 
        # next timestep of other particles
        dHdq_list = list(map(lambda i: self.dHdq(i), range(self.N)))
        dHdp_list = list(map(lambda i: self.dHdp(i), range(self.N)))
        prev_sign = 1 if self.particles[0].qy > 0 else -1
        for i,particle in enumerate(self.particles):
            if particle.fixed:  # dont move the particle
                continue
            dHdq_x, dHdq_y = dHdq_list[i]
            dHdp_x, dHdp_y = dHdp_list[i]
            particle.px = particle.px - self.dt*dHdq_x
            particle.py = particle.py - self.dt*dHdq_y
            particle.qx = particle.qx + self.dt*dHdp_x
            particle.qy = particle.qy + self.dt*dHdp_y
            if i == 0 and particle.qx < 0 :
                new_sign = 1 if particle.qy > 0 else -1
                if prev_sign != new_sign:
                    print("Aphelion",particle.qx-self.particles[1].qx)
                    

class System:
    def __init__(self, particles, integrator, tmax, dt, stepsperframe=2):
        self.particles = particles
        self.particle_positions = np.array([]) # todo define shape 
        self.integrator = integrator(particles, dt)
        self.tmax = tmax
        self.dt = dt
        self.t = 0
        self.stepsperframe = stepsperframe
        self.numframes = int(self.tmax/self.dt//self.stepsperframe)
        self.energies = []

    def step(self):
        self.integrator.integrate()
        self.t += self.dt
        # print(round(self.particles[0].qx,5), round(self.particles[0].py,5), end='\r')

    def H(self):
        T=sum(map(lambda body: body.kinetic_energy(), self.particles))
        U=0 # self-potential energy
        for i in range(len(self.particles)):
            for j in range(i+1, len(self.particles)):
                denom = norm(self.particles[i].qx-self.particles[j].qx, self.particles[i].qy-self.particles[j].qy)
                U += G*self.particles[i].m*self.particles[j].m/denom
        return T+U 

    def simulate(self):
        while self.t < self.tmax:
            self.step()

    def animate(self):
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
        d, = ax.plot([body.qx for body in self.particles],
                    [body.qy for body in self.particles], 'ro')
        # circle = plt.Circle((5, 5), 1, color='b', fill=False)
        # ax.add_artist(circle)
        # animation function.  This is called sequentially
        def nextframe(framenr):
            for frame in range(self.stepsperframe):
                self.step()
            self.energies.append(self.H())
            d.set_data([body.qx for body in self.particles],
                    [body.qy for body in self.particles])
            return d,
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, nextframe, frames=self.numframes, interval=20, blit=True)
        plt.show()

    def store_particle_positions(self):
        pass


if __name__ == "__main__":
    a=0.5
    R=5
    particle1 = Particle(R*(1-a), 0,0,1e8*math.sqrt((1+a)/(1-a)), m=1e8)
    particle2 = Particle(0,0,0,0, m=1e11)
    particles = [particle1, particle2]
    initial_state = InitialState(particles)
    system = System(initial_state.get_state(), SymplecticEuler, tmax=1, dt=0.0001,stepsperframe=1000)
    system.animate()
    
    plt.figure()
    plt.plot(system.energies, label="Energy over time")
    plt.legend()
    plt.show()
