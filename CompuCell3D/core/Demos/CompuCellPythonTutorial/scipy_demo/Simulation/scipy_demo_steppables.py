from PySteppables import *
from scipy import integrate
from numpy import exp


class ScipyDemoSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=100):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def step(self, mcs):
        f = lambda x: exp(-x ** 2)
        i = integrate.quad(f, 0, 1)

        print("integration result=", i)
