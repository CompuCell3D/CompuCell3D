from cc3d import CompuCellSetup
from .cellsort_2D_steering_steppables import ContactSteeringAndTemperature

CompuCellSetup.register_steppable(steppable=ContactSteeringAndTemperature(frequency=10))

CompuCellSetup.run()

