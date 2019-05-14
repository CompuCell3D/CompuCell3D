from cc3d.core.PySteppables import *


class FieldSecretionSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):

        fgf_field = self.field.FGF

        lmf_length = 1.0
        x_scale = 1.0
        y_scale = 1.0
        z_scale = 1.0
        # FOR HEX LATTICE IN 2D
        #         lmf_length=sqrt(2.0/(3.0*sqrt(3.0)))*sqrt(3.0)
        #         x_scale=1.0
        #         y_scale=sqrt(3.0)/2.0
        #         z_scale=sqrt(6.0)/3.0

        for cell in self.cell_list:
            # converting from real coordinates to pixels
            x_cm = int(cell.xCOM / (lmf_length * x_scale))
            y_cm = int(cell.yCOM / (lmf_length * y_scale))

            if cell.type == self.AMOEBA:
                fgf_field[x_cm, y_cm, 0] = 10.0

            elif cell.type == self.BACTERIA:
                fgf_field[x_cm, y_cm, 0] = 20.0
