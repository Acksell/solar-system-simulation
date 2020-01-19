import numpy as np
from constants import G

class Integrator:
    def __init__(self, particles, dt):
        self.particles = particles
        self.N = len(particles)
        self.dt = dt
        self.step = 0
    
    def integrate(self):
        # Should implement this method
        pass

    def dHdq(self, i, state=None):
        if state is None:
            state = np.array([[p.qx, p.qy, p.px, p.py] for p in self.particles])
        dHdq_x = 0
        dHdq_y = 0
        qx,qy,px,py = range(4) # indices for each property
        for i in range(self.N):
            for j in range(i+1, self.N):
                denom = ((state[i][qx] - state[j][qx])**2 + (state[i][qy] - state[j][qy])**2)**(3/2)
                dHdq_x += self.particles[i].m*self.particles[j].m*(state[i][qx] - state[j][qx])/denom
                dHdq_y += self.particles[i].m*self.particles[j].m*(state[i][qy] - state[j][qy])/denom
        return np.array([G*dHdq_x, G*dHdq_y])

    def dHdp(self, i, state=None):
        if state is None:
            state = np.array([[p.qx, p.qy, p.px, p.py] for p in self.particles])
        return np.array([state[i][2]/self.particles[i].m, state[i][3]/self.particles[i].m])

    def accel(self,i,state=None):
        return self.dHdq(i, state=state)/self.particles[i].m

class SymplecticEuler(Integrator):
    NAME="Symplectic Euler"
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
        self.step += 1

class Leapfrog(Integrator):
    NAME="Leapfrog"
    def integrate(self):
        accel_list = list(map(lambda i: self.accel(i), range(self.N)))

        for i, particle in enumerate(self.particles):
            if particle.fixed: continue
            accelx, accely = accel_list[i]
            if self.step == 0:
                # v_1/2
                particle.vx = particle.vx - accelx*self.dt/2
                particle.vy = particle.vy - accely*self.dt/2
            else:
                # v_i+1/2
                particle.vx = particle.vx - accelx*self.dt
                particle.vy = particle.vy - accely*self.dt
            particle.qx = particle.qx + particle.vx*self.dt
            particle.qy = particle.qy + particle.vy*self.dt
        self.step += 1

class RungeKutta4(Integrator):
    NAME="4th order Runge-Kutta"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = np.array([[p.qx, p.qy, p.px, p.py] for p in self.particles])
        self.zero_state = np.zeros((self.N, 4))

    def derivatives(self, state):
        acc = self.zero_state.copy() # Nx4 matrix [[qx, qy, px, py], ...]
        for i,row in enumerate(state):
            acc[i][0], acc[i][1] =  self.dHdp(i, state)  # q= dHdp, a vector/tuple
            acc[i][2], acc[i][3] = -self.dHdq(i, state)  # mv=p=-dHdq, a vector/tuple
        return acc

    def integrate(self):
        a=self.derivatives(self.state)*self.dt
        b=self.derivatives(self.state + a/2)*self.dt
        c=self.derivatives(self.state + b/2)*self.dt
        d=self.derivatives(self.state + c)*self.dt
        self.state += (a + 2*b + 2*c + d)/6
        for i,(qx,qy,px,py) in enumerate(self.state):
            self.particles[i].qx = qx
            self.particles[i].qy = qy
            self.particles[i].px = px
            self.particles[i].py = py
        self.step += 1