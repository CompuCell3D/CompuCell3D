from enum import Enum

FIELD_TYPES = (
    "CellField", "ConField", "ScalarField", "ScalarFieldCellLevel", "VectorField", "VectorFieldCellLevel", "CustomVis")

PLANES = ("xy", "xz", "yz")


class PlayerType(Enum):

    NEW = 'new'
    REPLAY = 'CMLResultReplay'


class ViewManagerType(Enum):

    REGULAR = 'Regular'
    REPLAY = 'CMLResultReplay'

    def __str__(self):
        return self.value
