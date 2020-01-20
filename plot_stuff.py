# BAD CODE THAT WAS PUT TOGETHER HASTILY TO MEET A DEADLINE

import math

import numpy as np
from matplotlib import pyplot as plt

from physics import Particle, System, InitialState
from Integrators import SymplecticEuler, Leapfrog, RungeKutta4, ForwardEuler
from constants import G

# a=0.5
R=152*1e9
Msol=1.9891e30
Mear=5.972e24
TMAX = 60*60*24*365*1

particle1 = Particle(q0x=R, q0y=0,p0x=0,p0y=29.29*1e3*Mear, m=Mear, name="Earth")
particle2 = Particle(0,0,0,0, m=Msol, fixed=False, name="Sun")
particles = [particle1, particle2]
initial_state = InitialState(particles)


def compare_integrators():
    integrators = [ForwardEuler, SymplecticEuler, Leapfrog, RungeKutta4]
    timesteps = [10000000,1000000,100000,10000,1000,100]

    errors = np.zeros((len(integrators), len(timesteps)))
    energy_drifts = np.zeros((len(integrators), len(timesteps)))
    energy_drifts_at_tmax = np.zeros((len(integrators), len(timesteps)))
    for i, integrator in enumerate(integrators):
        for j, dt in enumerate(timesteps):
            system = System(initial_state.get_state(), integrator, tmax=TMAX, dt=dt)
            
            system.simulate()
            print("abs error", system.error)
            print("rel error", system.error/R)
            errors[i][j] = system.error/R
            energy_drifts[i][j] = system.edrift
            energy_drifts_at_tmax[i][j] = abs(system.H(0) - system.initial_energies[0])/abs(system.initial_energies[0])
            print("edrift", system.edrift)
            # system.animate(boxlimits=2*R, interval=20)
            
            # system.plot_energies()
            # system.plot_trajectories()

    plt.figure()
    for i,error in enumerate(errors):
        plt.plot(timesteps,error, '-')
    plt.legend([integrator.NAME for integrator in integrators])
    plt.title("Relative errors after 1 orbit for different integrators and timesteps")
    plt.xlabel("dt")
    plt.ylabel("$\Delta x_{rel}$")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    print(errors)
    
    plt.figure()
    for i,edrift in enumerate(energy_drifts_at_tmax):
        plt.plot(timesteps, edrift, '-*')
    plt.legend([integrator.NAME for integrator in integrators])
    plt.title("Relative energy drifts after t=TMAX for different integrators and timesteps")
    plt.xlabel("dt")
    plt.ylabel("$\Delta E_{rel}$")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    print(energy_drifts_at_tmax)

    plt.figure()
    for i,edrift in enumerate(energy_drifts):
        plt.plot(timesteps, edrift, '-')
    plt.legend([integrator.NAME for integrator in integrators])
    plt.title("Relative energy drifts after passing qy=0 for different integrators and timesteps")
    plt.xlabel("dt")
    plt.ylabel("$\Delta E_{rel}$")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    print(energy_drifts)



def test_initial_conditions():
    system = System(initial_state.get_state(), Leapfrog, tmax=TMAX, dt=100, stepsperframe=10000)
    system.animate(boxlimits=1.3*R,interval=20)
    plt.figure()
    plt.plot(system.eccentricities)
    plt.title("Eccentricity over time")
    plt.show()
    system.plot_trajectories()
    system.plot_energy(0)

def asteroids():
    particle1 = Particle(q0x=R, q0y=0, p0x=0, p0y=29.29*1e3*Mear, m=Mear, name="Earth")
    particle2 = Particle(0,0,0,0, m=Msol, fixed=True, name="Sun")
    
    asteroid_v_angles = []
    earth_vangles=[]
    initial_vangles = np.array([45,40,30,20,15,10,5])*math.pi/180
    masses = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,2e1,5e1,1e2,3e2,5e2,7e2,1e3])
    # masses = np.array([1e0])

    ivangle = math.pi/2
    # mass=
    # norm(self.px, self.py)**2/self.m/2
    eccentricitiesEar = []
    eccentricitiesAst = []
    for i,mass in enumerate(masses):
        pabs = 10*1e3*mass*Mear
        timefactor=1.1
        plotstuff=False
        # if mass == 3e2:
        #     timefactor=1
        #     plotstuff=True
        particle3 = Particle(q0x=R*(1-0.001), q0y=0.1*R, p0x=pabs*math.cos(ivangle), p0y=-pabs*math.sin(ivangle), m=mass*Mear, name="Massive asteroid")
        # particle3 = Particle(q0x=-R, q0y=0, p0x=0, p0y=-29.29*1e3*Mear, m=Mear, name="Massive asteroid")

        parcls = [particle1, particle2, particle3]
        initstate = InitialState(parcls)
        system = System(initstate.get_state(), SymplecticEuler, tmax=TMAX*timefactor, dt=100, stepsperframe=400)
        system.simulate()
        # system.animate(boxlimits=1.5*R, interval=20)
        eccentricitiesEar.append(system.eccentricity(0))
        try:
            eccentricitiesAst.append(system.eccentricity(2))
        except ValueError:
            print("Negative asteroid eccentricity")
        if plotstuff:
            system.plot_trajectories(show=False,new_fig=not i)
        # system.plot_energies()
        vangleAst = math.atan(system.particles[2].py/system.particles[2].px)/math.pi*180
        vangleEar = math.atan(system.particles[0].py/system.particles[0].px)/math.pi*180
        earth_vangles.append(vangleEar)
        asteroid_v_angles.append(vangleAst)

    # particles[0].name="Undisturbed Earth"
    # initstate = InitialState(particles)
    # system = System(initstate.get_state(), SymplecticEuler, tmax=TMAX*timefactor, dt=100, stepsperframe=400)
    # system.simulate()
    # system.plot_trajectory(0,show=True,new_fig=False)

    plt.figure()
    plt.plot(masses, asteroid_v_angles, '-')
    plt.xscale("log")
    plt.title("Asteroid's velocity angle after Earth slingshot")
    plt.xlabel("$Asteroid\ mass\ [M⊕]$")
    plt.ylabel("Angle [degrees]")
    plt.show()

    plt.figure()
    plt.plot(masses, earth_vangles, '-')
    plt.title("Earth's velocity angle after massive asteroid slingshot")
    plt.xscale("log")
    plt.ylabel("Angle [degrees]")
    plt.xlabel("$Asteroid\ mass\ [M⊕]$")
    plt.show()

    plt.figure()
    plt.title("Earth's orbital eccentricity after massive asteroid slingshot")
    plt.plot(masses, eccentricitiesEar, '-')
    plt.xscale("log")
    plt.xlabel("$Asteroid\ mass\ [M⊕]$")
    plt.ylabel("$eccentricity$")
    plt.show()

    plt.figure()
    plt.title("Asteroids's orbital eccentricity after Earth slingshot")
    plt.plot(masses, eccentricitiesAst, '-')
    plt.xscale("log")
    plt.xlabel("$Asteroid\ mass\ [M⊕]$")
    plt.ylabel("$eccentricity$")
    plt.show()


    
# compare_integrators()
# test_initial_conditions()
asteroids()


