import re
from collections import OrderedDict


class SteppableTemplates:

    def __init__(self):
        self.steppableTemplatesDict = {}
        self.steppable_import_regex = OrderedDict(
            [
                ('1', re.compile('^from[\s]*cc3d\.core\.PySteppables[\s]*import[\s]*\*')),
                ('2', re.compile('^import[\s]*numpy[\s]*as[\s]*np')),
            ]
        )

        self.init_steppable_templates()

    def get_steppable_templates_dict(self):

        return self.steppableTemplatesDict

    def get_steppable_header_import_regex_list(self):
        return list(self.steppable_import_regex.values())

    def generate_steppable_import_header(self):
        """
        generates steppable imports
        :return:
        """
        # for each line of steppable import header we need to add regex in the constructor
        # because we check of existence of import lines before we insert imports into generated code
        return """
from cc3d.core.PySteppables import *
import numpy as np
"""

    def generate_steppable_code(self, steppable_name="GenericSteppable", frequency=1, steppable_type="Generic",
                                extra_fields=None):

        if extra_fields is None:
            extra_fields = []

        try:
            text = self.steppableTemplatesDict[steppable_type]

        except LookupError:

            return ""

        text = re.sub("STEPPABLENAME", steppable_name, text)

        text = re.sub("FREQUENCY", str(frequency), text)

        extra_fields_code = ''

        if "Scalar" in extra_fields:
            extra_fields_code += """
        self.create_scalar_field_py("SCALAR_FIELD_NAME")
"""

        if "ScalarCellLevel" in extra_fields:
            extra_fields_code += """
        self.create_scalar_field_cell_level_py("SCALAR_FIELD_CELL_LEVEL_NAME")
"""

        if "Vector" in extra_fields:
            extra_fields_code += """
        self.create_vector_field_py("VECTOR_FIELD_NAME")        
"""

        if "VectorCellLevel" in extra_fields:
            extra_fields_code += """
        self.create_vector_field_cell_level_py("VECTOR_FIELD_CELL_LEVEL_NAME")        
"""

        text = re.sub("EXTRAFIELDS", extra_fields_code, text)

        return text

    def generate_steppable_registration_code(self, steppable_name="GenericSteppable", frequency=1, steppable_file="",
                                             indentation_level=0, indentation_width=4):

        try:
            text = self.steppableTemplatesDict["SteppableRegistrationCode"]
        except LookupError:
            return ""

        text = re.sub("STEPPABLENAME", steppable_name, text)

        text = re.sub("STEPPABLEFILE", steppable_file, text)

        text = re.sub("FREQUENCY", str(frequency), text)

        # possible indentation of registration code - quite unlikely it wiil be needed

        if indentation_level < 0:
            indentation_level = 0

        text_lines = text.splitlines(True)

        for i in range(len(text_lines)):
            text_lines[i] = ' ' * indentation_width * indentation_level + text_lines[i]

        text = ''.join(text_lines)

        return text

    def init_steppable_templates(self):

        self.steppableTemplatesDict["SteppableRegistrationCode"] = """
        
from STEPPABLEFILE import STEPPABLENAME
CompuCellSetup.register_steppable(steppable=STEPPABLENAME(frequency=FREQUENCY))

"""

        self.steppableTemplatesDict["Generic"] = """        
class STEPPABLENAME(SteppableBasePy):
    def __init__(self, frequency=FREQUENCY):
        SteppableBasePy.__init__(self, frequency)
        EXTRAFIELDS

    def start(self):

        print("STEPPABLENAME: This function is called once before simulation")


    def step(self, mcs):
        print("STEPPABLENAME: This function is called every FREQUENCY MCS")

        for cell in self.cell_list:
            print("CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume)


    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return


"""

        self.steppableTemplatesDict["RunBeforeMCS"] = """
class STEPPABLENAME(RunBeforeMCSSteppableBasePy):
    def __init__(self, frequency=FREQUENCY):
        SteppableBasePy.__init__(self, frequency)
        EXTRAFIELDS

    def start(self):

        print("STEPPABLENAME: This function is called once before simulation")

    def step(self, mcs):
        print("STEPPABLENAME: This function is called every FREQUENCY MCS")
        print("STEPPABLENAME: This function is called before MCS i.e. pixel-copies take place for that MCS ")

        # typical use for this type of steppable is secretion  
        # uncomment lines  below and include Secretion plugin to make commented code work

        # attr_secretor = self.get_field_secretor("FIELD TO SECRETE")

        # for cell in self.cell_list:

            # if cell.type == 3:

                # attr_secretor.secreteInsideCell(cell,300)

                # attr_secretor.secreteInsideCellAtBoundary(cell,300)

                # attr_secretor.secreteOutsideCellAtBoundary(cell,500)

                # attr_secretor.secreteInsideCellAtCOM(cell,300)             

    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return


"""

        self.steppableTemplatesDict["Mitosis"] = """
class STEPPABLENAME(MitosisSteppableBase):
    def __init__(self,frequency=FREQUENCY):
        MitosisSteppableBase.__init__(self,frequency)

    def step(self, mcs):

        cells_to_divide=[]
        for cell in self.cell_list:
            if cell.volume>50:
                cells_to_divide.append(cell)

        for cell in cells_to_divide:

            self.divide_cell_random_orientation(cell)
            # Other valid options
            # self.divide_cell_orientation_vector_based(cell,1,1,0)
            # self.divide_cell_along_major_axis(cell)
            # self.divide_cell_along_minor_axis(cell)

    def update_attributes(self):
        # reducing parent target volume
        self.parent_cell.targetVolume /= 2.0                  

        self.clone_parent_2_child()            

        # for more control of what gets copied from parent to child use cloneAttributes function
        # self.clone_attributes(source_cell=self.parent_cell, target_cell=self.child_cell, no_clone_key_dict_list=[attrib1, attrib2]) 
        
        if self.parent_cell.type==1:
            self.child_cell.type=2
        else:
            self.child_cell.type=1

"""

        self.steppableTemplatesDict["ClusterMitosis"] = """
class STEPPABLENAME(MitosisSteppableClustersBase):

    def __init__(self, frequency=FREQUENCY):
        MitosisSteppableClustersBase.__init__(self, frequency)

    def step(self, mcs):

        for cell in self.cell_list:
            cluster_cell_list = self.get_cluster_cells(cell.clusterId)
            print("DISPLAYING CELL IDS OF CLUSTER ", cell.clusterId, "CELL. ID=", cell.id)
            for cell_local in cluster_cell_list:
                print("CLUSTER CELL ID=", cell_local.id, " type=", cell_local.type)

        mitosis_cluster_id_list = []
        for compartment_list in self.clusterList:
            # print( "cluster has size=",compartment_list.size())
            cluster_id = 0
            cluster_volume = 0
            for cell in CompartmentList(compartment_list):
                cluster_volume += cell.volume
                cluster_id = cell.clusterId

            # condition under which cluster mitosis takes place
            if cluster_volume > 250:
                # instead of doing mitosis right away we store ids for clusters which should be divide.
                # This avoids modifying cluster list while we iterate through it
                mitosis_cluster_id_list.append(cluster_id)

        for cluster_id in mitosis_cluster_id_list:

            self.divide_cluster_random_orientation(cluster_id)

            # # other valid options - to change mitosis mode leave one of the below lines uncommented
            # self.divide_cluster_orientation_vector_based(cluster_id, 1, 0, 0)
            # self.divide_cluster_along_major_axis(cluster_id)
            # self.divide_cluster_along_minor_axis(cluster_id)

    def update_attributes(self):
        # compartments in the parent and child clusters are
        # listed in the same order so attribute changes require simple iteration through compartment list
        compartment_list_parent = self.get_cluster_cells(self.parent_cell.clusterId)

        for i in range(len(compartment_list_parent)):
            compartment_list_parent[i].targetVolume /= 2.0
        self.clone_parent_cluster_2_child_cluster()
        
"""
