import sys, os


class BasicSimulationData():
    def __init__(self):
        self.sim = None
        self.fieldDim = None
        self.cell_types_used = None
        self.numberOfSteps = 0
        self.current_step = 0
