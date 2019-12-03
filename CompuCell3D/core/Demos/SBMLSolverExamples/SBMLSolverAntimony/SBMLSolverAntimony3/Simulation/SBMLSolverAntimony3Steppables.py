from cc3d.core.PySteppables import *


class SBMLSolverSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # Antimony model string: delta/notch/VEGFR2 regulation
        #    delta and notch decay into deltad and notchd
        #    delta and notch inhibit activated VEGFR (VEGFRa) by modifying its set point
        model_string = """model deltaNotch()
        # Model
        delta  => deltad; K1*(delta-k1)
        notch  => notchd; K2*(notch-k2)
        VEGFRa => VEGFRd; K3*(VEGFRa - k3/(1+sNotch*notch))
        
        # Initial conditions
        delta  = 0
        deltad = 1
        notch  = 0
        notchd = 1
        VEGFRa = 1
        VEGFRd = 0
        k1 = 0.00 # Delta set point
        K1 = 0.50 # Delta decay rate
        k2 = 0.00 # Notch set point
        K2 = 0.20 # Notch decay rate
        k3 = 1.00 # Base activated VEGFR set point
        K3 = 100.00 # VEGFR activation rate
        sNotch = 100 # Notch inhibition of activated VEGFR
        end"""

        options = {'relative': 1e-10, 'absolute': 1e-12}
        self.set_sbml_global_options(options)
        step_size = 1e-2

        # Apply model string to cells
        for cell in self.cellList:
            self.add_antimony_to_cell(model_string=model_string, model_name='dp', cell=cell, step_size=step_size)

    def step(self, mcs):
        self.timestep_sbml()


# Display SBML values
class IdFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.create_scalar_field_cell_level_py("delta")
        self.create_scalar_field_cell_level_py("VEGFR")
        self.create_scalar_field_cell_level_py("notch")

    def step(self, mcs):
        delta = self.field.delta
        vegfr = self.field.VEGFR
        notch = self.field.notch

        for cell in self.cell_list:
            delta[cell] = cell.sbml.dp['delta']
            vegfr[cell] = cell.sbml.dp['VEGFRa']
            notch[cell] = cell.sbml.dp['notch']


# VEGFR-dependent VEGF consumption and delta activation
class SecretionSteppable(SecretionBasePy):
    def __init(self, frequency=1):
        SecretionBasePy.__init__(self, frequency)

        self.vegf = None

    def start(self):

        self.vegf = CompuCell.getConcentrationField(self.simulator, "VEGF")

    def step(self, mcs):

        # VEGF consumption rate
        consumption_base = 5e-3

        # Conversion coefficient from consumed VEGF to delta per VEGFRa
        vegf_2_delta = 1e0

        vegf = self.vegf

        for cell in self.cell_list:
            this_delta = cell.sbml.dp['delta']
            this_vegfra = cell.sbml.dp['VEGFRa']

            pixel_list = CellPixelList(self.pixelTrackerPlugin, cell)
            vegf_consumed = 0
            for pixelData in pixel_list:
                pt = pixelData.pixel
                field_value = vegf.get(pt)
                this_veg_fconsumed = field_value * consumption_base
                vegf_consumed += this_veg_fconsumed
                vegf.set(pt, field_value - this_veg_fconsumed)

            delta_activated = this_vegfra * vegf_consumed * vegf_2_delta
            new_delta = min(this_delta + delta_activated, 1)
            cell.sbml.dp['delta'] = new_delta


# Delta-notch signaling
class DeltaNotchNeighborSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):

        delta_2_notch = 1e-4  # Conversion coefficient from delta to notch per delta per common surface area
        delta_threshold = 0.75  # Delta signals notch in neighbors when above this value

        for cell in self.cell_list:
            notch_2_add = 0
            this_notch = cell.sbml.dp['notch']
            for neighbor_tuple in self.get_cell_neighbor_data_list(cell):
                neighbor_cell = neighbor_tuple[0]
                common_area = neighbor_tuple[1]
                if neighbor_cell:
                    neighbor_delta = neighbor_cell.sbml.dp['delta']
                    if neighbor_delta > delta_threshold:
                        notch_2_add += neighbor_delta * common_area * delta_2_notch

            new_notch = min(this_notch + notch_2_add, 1)
            cell.sbml.dp['notch'] = new_notch


# VEGFR-regulated chemotaxis
class NotchChemotaxisSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cell_list:
            cd = self.chemotaxisPlugin.addChemotaxisData(cell, "VEGF")
            cd.setLambda(0.0)
            cd.assignChemotactTowardsVectorTypes([self.MEDIUM, self.STALKCELL])

    def step(self, mcs):
        # Chemotaxis Lagrange multiplier
        lambda_base = 5e3

        # Chemotaxis saturation
        chemo_sat = 1e0
        for cell in self.cell_list:
            cd = self.chemotaxisPlugin.getChemotaxisData(cell, "VEGF")
            this_vegfra = cell.sbml.dp['VEGFRa']
            cd.setLambda(lambda_base * this_vegfra)
            cd.setSaturationCoef(chemo_sat)
            cd.assignChemotactTowardsVectorTypes([self.MEDIUM, self.STALKCELL])
