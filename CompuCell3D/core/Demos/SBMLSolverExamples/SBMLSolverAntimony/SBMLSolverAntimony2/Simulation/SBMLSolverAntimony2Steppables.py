from cc3d.core.PySteppables import *


class SBMLSolverSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # Antimony model string: cell type 1
        model_string_type1 = """model type1()
        # Model
        S1 =>  S2; k1*S1
        
        # Initial conditions
        S1 = 0
        S2 = 1
        k1 = 1
        end"""

        # Antimony model string: cell type 2
        model_string_type2 = """model type2()
        # Model
        S2 =>  S1; k2*S2
        
        # Initial conditions
        S1 = 0
        S2 = 0
        k2 = 1
        end"""

        # adding options that setup SBML solver integrator
        # these are optional but useful when encountering integration instabilities

        options = {'relative': 1e-10, 'absolute': 1e-12}
        self.set_sbml_global_options(options)
        step_size = 0.001

        # Apply model strings to cell types
        self.add_antimony_to_cell_types(model_string=model_string_type1, model_name='dpType1', cell_types=[self.TYPE1],
                                        step_size=step_size)
        self.add_antimony_to_cell_types(model_string=model_string_type2, model_name='dpType2', cell_types=[self.TYPE2],
                                        step_size=step_size)

    def step(self, mcs):
        self.timestep_sbml()

    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return


class SecretionSteppable(SecretionBasePy):
    def __init(self, frequency=1):
        SecretionBasePy.__init__(self, frequency)

    def step(self, mcs):
        consume_s1 = 1
        consume_s2 = 1
        secrete_s1 = 1
        secrete_s2 = 1

        field1 = self.field.Field1
        field2 = self.field.Field2

        for cell in self.cell_list_by_type(self.TYPE1):
            this_cell_s1 = cell.sbml.dpType1['S1']
            this_cell_s2 = cell.sbml.dpType1['S2']
            cell_volume = cell.volume

            if this_cell_s2 > 0.75:
                this_secrete_s2 = secrete_s2
            else:
                this_secrete_s2 = 0

            pixel_list = CellPixelList(self.pixelTrackerPlugin, cell)
            sbml_values = cell.sbml.dpType1.values()
            s1_consumed = 0
            for pixel_data in pixel_list:
                pt = pixel_data.pixel
                field_value = field1.get(pt)
                s1_consumed += field_value * consume_s1

            s2_secreted = this_cell_s2 * this_secrete_s2

            cell.sbml.dpType1['S1'] = this_cell_s1 + s1_consumed
            cell.sbml.dpType1['S2'] = this_cell_s2 - s2_secreted

            for pixel_data in pixel_list:
                pt = pixel_data.pixel

                field1_val = field1.get(pt) - s1_consumed / cell_volume
                field2_val = field2.get(pt) + s2_secreted / cell_volume
                field1.set(pt, field1_val)
                field2.set(pt, field2_val)

        for cell in self.cell_list_by_type(self.TYPE2):
            this_cell_s1 = cell.sbml.dpType2['S1']
            this_cell_s2 = cell.sbml.dpType2['S2']
            cell_volume = cell.volume

            if this_cell_s1 > 0.75:
                this_secrete_s1 = secrete_s1
            else:
                this_secrete_s1 = 0

            pixel_list = CellPixelList(self.pixelTrackerPlugin, cell)
            s2_consumed = 0
            for pixel_data in pixel_list:
                pt = pixel_data.pixel
                field_value = field2.get(pt)
                s2_consumed += field_value * consume_s2

            S1_secreted = this_cell_s1 * this_secrete_s1

            cell.sbml.dpType2['S1'] = this_cell_s1 - S1_secreted
            cell.sbml.dpType2['S2'] = this_cell_s2 + s2_consumed

            for pixel_data in pixel_list:
                pt = pixel_data.pixel

                field1_val = field1.get(pt) + S1_secreted / cell_volume
                field2_val = field2.get(pt) - s2_consumed / cell_volume
                field1.set(pt, field1_val)
                field2.set(pt, field2_val)


# Demo: accessing SBML values for further manipulation/coupling with other components
class IdFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.create_scalar_field_cell_level_py("IdFieldS1")
        self.create_scalar_field_cell_level_py("IdFieldS2")

    def step(self, mcs):
        id_field_s1 = self.field.IdFieldS1
        id_field_s2 = self.field.IdFieldS2

        for cell in self.cell_list_by_type(self.TYPE1):
            id_field_s1[cell] = cell.sbml.dpType1['S1']
            id_field_s2[cell] = cell.sbml.dpType1['S2']
        for cell in self.cell_list_by_type(self.TYPE2):
            id_field_s1[cell] = cell.sbml.dpType2['S1']
            id_field_s2[cell] = cell.sbml.dpType2['S2']
