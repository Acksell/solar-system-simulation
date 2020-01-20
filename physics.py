# Final project in SI1336

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

from Integrators import SymplecticEuler, Leapfrog, RungeKutta4
from constants import G
from helpers import norm

class Particle:
    def __init__(self,q0x,q0y,p0x,p0y, m, fixed=False,name=""):
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
        self.name=name

        self.qx_list = []
        self.qy_list = []
        self.prev_sign = None

    @property
    def vx(self):
        return self.px/self.m
    
    @property
    def vy(self):
        return self.py/self.m
    
    @vx.setter
    def vx(self, value):
        self.px = value*self.m
    
    @vy.setter
    def vy(self, value):
        self.py = value*self.m

    def store_position(self):
        self.qx_list.append(self.qx)
        self.qy_list.append(self.qy)

    def kinetic_energy(self):
        return norm(self.px, self.py)**2/self.m/2

    def angular_momentum(self, origin=(0,0)):
        return np.cross([self.qx,self.qy,0],[self.px,self.py,0])

    def __str__(self):
        return "<Particle(qx={}\tqy={}\tvx={}\tvy={},\tm={})>".format(round(self.qx,2),round(self.qy,2),round(self.vx,2),round(self.vy,2),self.m)
    __repr__ = __str__

    def copy(self):
        return Particle(self.qx, self.qy, self.px, self.py, self.m, fixed=self.fixed, name=self.name)

class InitialState:
    def __init__(self, initialparticles):
        self.initialparticles = initialparticles

    def get_state(self):
        particles = []
        for particle in self.initialparticles:
            particles.append(particle.copy())
        return particles

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
        self.energies = [[] for p in self.particles]
        self.eccentricities = []
        self.error = 0
        self.edrift = 0
        self.initial_Etot = self.Etot()
        self.initial_energies = [self.H(i) for i in range(len(self.particles))]

    def step(self):
        self.integrator.integrate()
        self.t += self.dt
        if 0.8 < self.t/self.tmax < 1.4:
            if self.particles[0].prev_sign is not None:
                if self.particles[0].prev_sign != np.sign(self.particles[0].qy):
                    self.error = abs(self.particles[0].qx - self.particles[0].q0x)
                    print("found error at qy=",self.particles[0].qy)
                    self.edrift = abs(self.H(0) - self.initial_energies[0])/abs(self.initial_energies[0])
                    print("found edrift",self.edrift, self.H(0) - self.initial_energies[0],self.initial_energies[0])
            self.particles[0].prev_sign = np.sign(self.particles[0].qy)
        if self.t % 10000 == 0:
            for i in range(len(self.particles)):
                self.energies[i].append(self.H(i) - self.initial_energies[i])
            try:
                ecc = self.eccentricity(0)
                # ecc2= self.eccentricity(2)
                self.eccentricities.append(ecc)
            except ValueError:
                print("[",round(self.t/self.tmax,3),"]","Hyperbolic")
            else:
                print("[",round(self.t/self.tmax,3),"]", ecc)
            for p in self.particles:
                if p.name.lower() != "sun": p.store_position()

    def H(self, idx=None, heliocentric=False):
        if idx is None:
            T=sum(map(lambda body: body.kinetic_energy(), self.particles))
        else:
            T=self.particles[idx].kinetic_energy()
        U=0 # self-potential energy
        for i in range(len(self.particles)):
            if idx is not None and idx != i: continue  # skip if not specific particle
            for j in range(i+1, len(self.particles)):
                if heliocentric and self.particles[j].name.lower() != "sun": continue
                denom = norm(self.particles[i].qx-self.particles[j].qx, self.particles[i].qy-self.particles[j].qy)
                U -= G*self.particles[i].m*self.particles[j].m/denom
        return T+U 

    def Etot(self, diff=False):
        if diff: # should plot the difference to the initial energy instead
            Efunc = lambda i: self.H(i)-self.initial_Etot
        else:
            Efunc = self.H
        return sum(map(Efunc, range(len(self.particles))))

    def reduced_mass(self):
        one_over_m = sum(map(lambda body: 1/body.m, self.particles))
        return 1/one_over_m

    def total_angular_momentum(self):
        #vector Ltot
        Ltot = sum(map(lambda body: body.angular_momentum(),self.particles))
        sgn = 1 if Ltot[2] > 0 else -1
        return sgn*np.linalg.norm(Ltot)

    def eccentricity(self, i):
        Ltot = np.linalg.norm(self.particles[i].angular_momentum())
        Etot = self.H(i, heliocentric=True)
        reduced_mass = 1/(1/self.particles[1].m + 1/self.particles[i].m)
        return math.sqrt(1 + 2*Etot*Ltot**2/(reduced_mass*(G*self.particles[i].m*self.particles[1].m)**2))

    def simulate(self):
        while self.t < self.tmax:
            self.step()

    def animate(self, boxlimits=10, interval=20):
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(-boxlimits, boxlimits), ylim=(-boxlimits, boxlimits))
        d, = ax.plot([body.qx for body in self.particles],
                    [body.qy for body in self.particles], 'ro')
        # circle = plt.Circle((5, 5), 1, color='b', fill=False)
        # ax.add_artist(circle)
        # animation function.  This is called sequentially
        def nextframe(framenr):
            for frame in range(self.stepsperframe):
                self.step()
            d.set_data([body.qx for body in self.particles],
                    [body.qy for body in self.particles])
            return d,
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, nextframe, frames=self.numframes, interval=interval, blit=True)
        plt.show()
        
    def plot_energies(self):
        plt.figure()
        for i,p in enumerate(self.particles):
            plt.plot(self.energies[i], label=p.name)
        etot = list(map(sum, zip(*self.energies)))
        plt.plot(etot, label="Total energy")
        plt.title("Energy over time," + self.integrator.NAME)
        plt.legend()
        plt.show()

    def plot_energy(self, i):
        plt.figure()
        plt.plot(self.energies[i], label=self.particles[i].name)
        plt.title("Energy over time, " + self.integrator.NAME)
        plt.legend()
        plt.show()

    def plot_trajectories(self,show=True,new_fig=True):
        if new_fig: plt.figure()
        for i, p in enumerate(self.particles):
            if p.name.lower() == "sun":
                plt.plot([0],[0],'or', label=p.name) # sun at center
            else:
                plt.plot(p.qx_list, p.qy_list, '-', label=p.name)
        plt.title("Solar system trajectories")
        plt.legend()
        if show:
            plt.show()

    def plot_trajectory(self,i,show,new_fig):
        if new_fig: plt.figure()
        plt.plot(self.particles[i].qx_list, self.particles[i].qy_list, '-', label=self.particles[i].name)
        plt.title("Solar system trajectories")
        if show:
            plt.legend(loc="lower left")
            plt.show()


if __name__ == "__main__":
    a=0.5
    R=152*1e9
    Msol=1.9891e30
    Mear=5.972e24

    particle1 = Particle(q0x=R, q0y=0,p0x=0,p0y=29.29*1e3*Mear, m=Mear)
    particle2 = Particle(0,0,0,0, m=Msol, fixed=False)
    particles = [particle1, particle2]
    initial_state = InitialState(particles)

    integrators = [SymplecticEuler, Leapfrog, RungeKutta4]
    timesteps = [10000,1000,100,10]

    errors = np.zeros((len(integrators), len(timesteps)))
    
    TMAX = 60*60*24*365*1.1
    for i, integrator in enumerate(integrators):
        for j, dt in enumerate(timesteps):
            system = System(initial_state.get_state(), integrator, tmax=TMAX, dt=dt)
            
            system.simulate()
            print("abs error", system.error)
            print("rel error", system.error/R)
            errors[i][j] = system.error/R
            # system.animate(boxlimits=2*R, interval=20)
            
            plt.figure()
            plt.plot(system.energies, label="")
            plt.legend(["Energy over time," + integrator.NAME])
            plt.show()

            plt.figure()
            plt.plot(system.particles[0].qx_list, system.particles[0].qy_list, '-')
            plt.plot([0],[0],'or')
            plt.plot(system.particles[1].qx_list, system.particles[1].qy_list, '-')
            plt.title("Sun-Earth system")
            plt.legend(["Earth","Sun"])
            plt.show()
    
    plt.figure()
    for i,error in enumerate(errors):
        plt.plot(timesteps,error, '-')
    plt.legend([integrator.NAME for integrator in integrators])
    plt.title("Errors after 1 orbit for different integrators and timestep")
    plt.xlabel("dt")
    plt.ylabel("$\Delta x_{rel}$")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    print(errors)
    

