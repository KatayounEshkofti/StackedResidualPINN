#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
from scipy.stats import truncnorm

#%%###########################################################################
#                       1) GODUNOV SIMULATION CODE
############################################################################
def flux(Vf, greenshield=True):
    if greenshield:
        rhoc = 0.5
        def f(rho):
            return Vf * rho * (1 - rho)
    else:
        rhoc = 0.4
        def f(rho):
            return Vf*rho*(rho <= rhoc) + Vf*rhoc*(rho - 1)/(rhoc - 1)*(rho > rhoc)
    return (f, rhoc)

class PhysicsSim:
    def __init__(self, L, Nx, Tmax, Vf=1, gamma=0.05, greenshield=True):
        self.Nx = Nx
        self.L = L
        self.Tmax = Tmax
        self.update(Vf, gamma)
        self.greenshield = greenshield

    def update(self, Vf, gamma):
        self.Vf = Vf
        self.gamma = gamma
        self.deltaX = self.L / self.Nx
        if gamma > 0:
            self.deltaT = 0.8 * min(self.deltaX / Vf, self.deltaX**2/(2*gamma))
        else:
            self.deltaT = 0.8 * self.deltaX / Vf
        self.Nt = int(np.ceil(self.Tmax / self.deltaT))

class ProbeVehicles:
    def __init__(self, sim, xiPos, xiT):
        self.sim = sim
        self.Nxi = len(xiPos)
        # Convert positions/time in [x, t] to integer grid indices
        self.xi = [np.array([int(xiPos[i]*sim.Nx/sim.L)], dtype=int) for i in range(self.Nxi)]
        self.xiT = [np.array([int(xiT[i]*sim.Nt/sim.Tmax)], dtype=int) for i in range(self.Nxi)]
        self.xiArray = [np.array([xiPos[i]]) for i in range(self.Nxi)]
        self.xiTArray = [np.array([xiT[i]]) for i in range(self.Nxi)]

    def update(self, z, n):
        for j in range(self.Nxi):
            if (self.xi[j][-1] >= self.sim.Nx) or (n*self.sim.Tmax/self.sim.Nt < self.xiTArray[j][-1]):
                continue
            new_xiPos = self.xiArray[j][-1] + self.sim.deltaT * self.speed(z[self.xi[j][-1]])
            self.xiArray[j] = np.append(self.xiArray[j], new_xiPos)
            self.xiTArray[j] = np.append(self.xiTArray[j], n*self.sim.Tmax/self.sim.Nt)

            new_xi = int(new_xiPos * self.sim.Nx / self.sim.L)
            self.xi[j] = np.append(self.xi[j], new_xi)
            self.xiT[j] = np.append(self.xiT[j], n)

    def speed(self, z):
        if z > 0:
            f, _ = flux(self.sim.Vf, greenshield=self.sim.greenshield)
            return f(z)/z
        else:
            return self.sim.Vf

    def getMeasurements(self, z):
        xMeasurements = [np.empty((0, self.Nxi))]*self.Nxi
        tMeasurements = [np.empty((0, self.Nxi))]*self.Nxi
        zMeasurements = [np.empty((0, self.Nxi))]*self.Nxi
        vMeasurements = [np.empty((0, self.Nxi))]*self.Nxi
        for j in range(self.Nxi):
            tMeasurements[j] = self.xiTArray[j][0:-1]
            xMeasurements[j] = self.xiArray[j][0:-1]
            for n in self.xiT[j][0:-1]:
                newDensity = z[self.xi[j][n], self.xiT[j][n]]
                zMeasurements[j] = np.append(zMeasurements[j], newDensity)
                vMeasurements[j] = np.append(vMeasurements[j], self.speed(newDensity))
        return (xMeasurements, tMeasurements, zMeasurements, vMeasurements)

    def plot(self):
        for j in range(self.Nxi):
            plt.plot(self.xiTArray[j], self.xiArray[j], c='k')

class BoundaryConditions:
    def __init__(self, sim, minZ0, maxZ0, rhoBar=-1, rhoSigma=0, sinePuls=15):
        self.minZ0 = minZ0
        self.maxZ0 = maxZ0
        self.sinePuls = sinePuls
        self.sim = sim
        Tx = 0.4
        Tt = 0.20

        if rhoBar == -1 and rhoSigma == 0:
            self.randomGaussian = False
        else:
            self.randomGaussian = True

        if self.randomGaussian:
            self.X = truncnorm((minZ0 - rhoBar) / rhoSigma,
                               (maxZ0 - rhoBar) / rhoSigma,
                               loc=rhoBar, scale=rhoSigma)
            self.Npoints = [int(np.ceil(sim.Tmax/Tt)), int(np.ceil(sim.L/Tx))]
        else:
            Npoints = int(np.ceil(sim.Tmax*sinePuls/(2*np.pi)))
            self.randomT = np.sort(np.random.randint(0, sim.Nt, (2, Npoints)))
            self.randomT[0,-1] = sim.Nt
            self.randomT[1,-1] = sim.Nt
            self.randomValues = minZ0 + np.random.rand(2, Npoints)*(maxZ0 - minZ0)

    def getZ0(self):
        if self.randomGaussian:
            points = self.sim.L * lhs(1, samples=self.Npoints[1])
            points = (points/self.sim.deltaX).astype(int)
            points = np.sort(points.reshape((self.Npoints[1],)))
            points = np.append(points, self.sim.Nx)
            z0Values = self.X.rvs((self.Npoints[1] + 1,))
            z0 = np.ones((points[0], 1))
            for i in range(self.Npoints[1]):
                z0 = np.vstack((z0,
                                np.ones((points[i+1] - points[i], 1))
                                * z0Values[i+1]))
        else:
            Nx = self.sim.Nx
            L = self.sim.L
            averageSine = (self.maxZ0 + self.minZ0)/2
            amplitudeSine = (self.maxZ0 - self.minZ0)/2
            Nx1 = int(np.floor(1*Nx/L))
            Nx2 = int(np.floor(0.2*Nx/L))
            Nx3 = int(np.floor(3.1*Nx/L))
            Nx4 = Nx - Nx1 - Nx2 - Nx3
            angleSine = np.vstack(self.sinePuls*np.sqrt(np.arange(Nx3)*L/Nx))
            z0 = np.concatenate((
                averageSine*np.ones((Nx3, 1)) + amplitudeSine*np.cos(angleSine),
                np.ones((Nx2, 1))*self.minZ0,
                (self.minZ0+self.maxZ0)/2*np.ones((Nx1, 1)),
                (1*self.minZ0 + 4*self.maxZ0)/5*np.ones((Nx4, 1))
            ), axis=0)
        return z0

    def getZbottom(self):
        if self.randomGaussian:
            points = self.sim.Tmax*lhs(1, samples=self.Npoints[0])
            points = (points/self.sim.deltaT).astype(int)
            points = np.sort(points.reshape((self.Npoints[0],)))
            points = np.append(points, self.sim.Nt)
            zinValues = self.X.rvs((self.Npoints[0]+1, ))
            zin = np.ones((points[0], 1))*zinValues[0]
            for i in range(self.Npoints[0]):
                zin = np.vstack((zin,
                                 np.ones((points[i+1] - points[i], 1))
                                 * zinValues[i+1]))
        else:
            Nt = self.sim.Nt
            Tmax = self.sim.Tmax
            angleCos = np.vstack(self.sinePuls*np.sqrt(np.arange(Nt)*Tmax/Nt))
            zin = np.ones((Nt,1))*self.minZ0 + (self.maxZ0 - self.minZ0)*(np.cos(angleCos)+1)/2

            zin = np.ones((self.randomT[1,0], 1))*self.randomValues[1,0]
            for i in range(self.randomT.shape[1]-1):
                zin = np.vstack((zin,
                                 np.ones((self.randomT[0,i+1]-self.randomT[0,i], 1))
                                 * self.randomValues[0,i+1]))
        return zin

    def getZtop(self):
        if self.randomGaussian:
            points = self.sim.Tmax*lhs(1, samples=self.Npoints[0])
            points = (points/self.sim.deltaT).astype(int)
            points = np.sort(points.reshape((self.Npoints[0],)))
            points = np.append(points, self.sim.Nt)
            zinValues = self.X.rvs((self.Npoints[0]+1, ))
            zinValues = np.ones((self.Npoints[0]+1, ))
            zin = np.ones((points[0], 1))*zinValues[0]
            for i in range(self.Npoints[0]):
                zin = np.vstack((zin,
                                 np.ones((points[i+1]-points[i], 1))
                                 * zinValues[i+1]))
        else:
            Nt = self.sim.Nt
            Tmax = self.sim.Tmax
            angleCos = np.vstack(6*np.arange(Nt)*Tmax/Nt)
            zin = np.ones((Nt,1))*self.maxZ0 + (self.maxZ0 - self.minZ0)*(np.cos(angleCos)-1)/4

            zin = np.ones((self.randomT[0,0], 1))*self.randomValues[0,0]
            for i in range(self.randomT.shape[1]-1):
                zin = np.vstack((zin,
                                 np.ones((self.randomT[0,i+1]-self.randomT[0,i], 1))
                                 * self.randomValues[0,i+1]))
        return zin

class SimuGodunov:
    def __init__(
        self, Vf, gamma, xiPos, xiT, zMin=0, zMax=1, L=5, Tmax=2,
        Nx=300, rhoBar=-1, rhoSigma=0, greenshield=True
    ):
        self.sim = PhysicsSim(L, Nx, Tmax, Vf, gamma, greenshield)
        bc = BoundaryConditions(self.sim, zMin, zMax, rhoBar, rhoSigma)
        self.z0 = bc.getZ0()
        self.zBottom = bc.getZbottom()
        self.zTop = bc.getZtop()
        self.pv = ProbeVehicles(self.sim, xiPos, xiT)
        self.zMax = zMax
        self.zMin = zMin
        self.t = np.linspace(0, self.sim.Tmax, self.sim.Nt)
        self.x = np.linspace(0, self.sim.L, self.sim.Nx)

    def g(self, u, v):
        f, rhoc = flux(self.sim.Vf, greenshield=self.sim.greenshield)
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        if u > v:
            if v >= rhoc:
                return f(v)
            elif u <= rhoc:
                return f(u)
            else:
                return f(rhoc)
        else:
            return min(f(u), f(v))

    def simulation(self):
        Nx = self.sim.Nx
        Nt = self.sim.Nt
        deltaX = self.sim.deltaX
        deltaT = self.sim.deltaT
        Vf = self.sim.Vf
        gamma = self.sim.gamma

        z = np.zeros((Nx, Nt))
        # Initialize
        for i in range(Nx):
            z[i, 0] = self.z0[i].item()

        for n in range(1, Nt):
            z[0, n] = np.clip(self.zTop[n], self.zMin, self.zMax)
            for i in range(1, Nx - 1):
                if gamma > 0:
                    z[i, n] = z[i, n-1] + deltaT * (
                        gamma*(z[i-1, n-1] - 2*z[i, n-1] + z[i+1, n-1])/deltaX**2
                        - Vf*(1 - 2*z[i, n-1])*(z[i+1, n-1] - z[i-1, n-1])/(2*deltaX)
                    )
                else:
                    gpdemi = self.g(z[i, n-1], z[i+1, n-1])
                    gmdemi = self.g(z[i-1, n-1], z[i, n-1])
                    z[i, n] = z[i, n-1] - deltaT*(gpdemi - gmdemi)/deltaX

                z[i, n] = np.clip(z[i, n], self.zMin, self.zMax)

            z[-1, n] = np.clip(self.zBottom[n], self.zMin, self.zMax)
            self.pv.update(z[:, n], n)

        self.z = z
        return z

    def getAxisPlot(self):
        return (self.x, self.t)

    def plot(self):
        z = self.z
        fig = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(self.t, self.x)
        plt.pcolor(X, Y, z, vmin=0.0, vmax=1.0, cmap='rainbow', shading='auto', rasterized=True)
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.xlim(0, self.sim.Tmax)
        plt.ylim(0, self.sim.L)
        plt.colorbar()
        plt.tight_layout()
        self.pv.plot()
        plt.show()

    def getMeasurements(self, selectedPacket=-1, totalPacket=-1, noise=False):
        x_true, t, rho_true, v_true = self.pv.getMeasurements(self.z)
        Nxi = len(x_true)

        x_selected = []
        t_selected = []
        rho_selected = []
        v_selected = []
        for k in range(Nxi):
            Nt_ = t[k].shape[0]
            if totalPacket == -1:
                totalPacket = Nt_
            if selectedPacket <= 0:
                selectedPacket = totalPacket
            elif selectedPacket < 1:
                selectedPacket = int(np.ceil(totalPacket*selectedPacket))

            nPackets = int(np.ceil(Nt_/totalPacket))
            toBeSelected = np.empty((0,1), dtype=int)
            for i in range(nPackets):
                randomPackets = np.arange(i*totalPacket, min((i+1)*totalPacket, Nt_), dtype=int)
                np.random.shuffle(randomPackets)
                if selectedPacket > randomPackets.shape[0]:
                    toBeSelected = np.append(toBeSelected, randomPackets[0:-1])
                else:
                    toBeSelected = np.append(toBeSelected, randomPackets[0:selectedPacket])
            toBeSelected = np.sort(toBeSelected)

            if noise:
                noise_trajectory = np.random.normal(0, 1.5, Nt_)/1000
                noise_trajectory = np.cumsum(noise_trajectory.reshape(-1,), axis=0)
                noise_meas = np.random.normal(0, 0.02, Nt_).reshape(-1,)
            else:
                noise_trajectory = np.array([0]*Nt_)
                noise_meas = np.array([0]*Nt_)

            x_sel_k = x_true[k][toBeSelected] + noise_trajectory[toBeSelected]
            rho_temp = rho_true[k][toBeSelected] + noise_meas[toBeSelected]
            rho_temp = np.maximum(np.minimum(rho_temp, 1.0), 0.0)

            x_selected.append(torch.tensor(x_sel_k.reshape(-1,1), dtype=torch.float32))
            t_selected.append(torch.tensor(t[k][toBeSelected].reshape(-1,1), dtype=torch.float32))
            rho_selected.append(torch.tensor(rho_temp.reshape(-1,1), dtype=torch.float32))
            v_selected.append(torch.tensor(v_true[k][toBeSelected].reshape(-1,1), dtype=torch.float32))

        return t_selected, x_selected, rho_selected, v_selected

    def getDatas(self, x, t):
        X = (x/self.sim.deltaX).astype(int)
        T = (t/self.sim.deltaT).astype(int)
        return self.z[X, T]

    def getPrediction(self, tf, Nexp=10, wMax=30, Amax=1, Amin=0):
        Nplus = int((tf-self.sim.Tmax)/self.sim.deltaT)
        wRand = wMax*np.random.rand(Nexp, 2)
        Arand = Amin + (Amax-Amin)*np.random.rand(Nexp, 2)
        Brand = Amin + (Amax-Amin)*np.random.rand(Nexp, 2)
        Crand = Amin + (Amax-Amin)*np.random.rand(Nexp, 2)

        t = np.linspace(self.sim.Tmax, tf, Nplus)
        boundaryValues = np.zeros((Nplus, Nexp*2))
        for i in range(Nexp):
            boundaryValues[:,2*i]   = Crand[i,0] + Arand[i,0]*np.sin(wRand[i,0]*t) + Brand[i,0]*np.cos(wRand[i,0]*t)
            boundaryValues[:,2*i+1] = Crand[i,1] + Arand[i,1]*np.sin(wRand[i,1]*t) + Brand[i,1]*np.cos(wRand[i,1]*t)
        boundaryValues = np.clip(boundaryValues, 0.0, 1.0)
        return (t, boundaryValues)