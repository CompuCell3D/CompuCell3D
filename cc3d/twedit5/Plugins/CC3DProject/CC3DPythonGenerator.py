import os.path


def generate_configure_simulation_header():
    configureSimLines = '''
def configure_simulation():

    from cc3d.core.XMLUtils import ElementCC3D
    
'''

    return configureSimLines


def generate_configure_sim_fcn_body(root_element, output_file_name):
    # note the XML root element generated using C++ xml-to-python converter is called RootElementNameElmnt
    # here it will be CompuCell3DElmnt

    configure_sim_file_name = str(output_file_name)

    # _rootElement.saveXMLInPython(configure_sim_file_name)

    python_xml_file = open(configure_sim_file_name, 'w')

    python_xml_file.write('%s' % root_element.getXMLAsPython())

    python_xml_file.close()

    configure_sim_lines = generate_configure_simulation_header()

    configure_sim_file = open(configure_sim_file_name, "r")

    configure_sim_body = configure_sim_file.read()

    configure_sim_file.close()

    configure_sim_lines += configure_sim_body

    configure_sim_lines += '''

    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)    

    '''

    configure_sim_lines += '\n'

    os.remove(configure_sim_file_name)

    return configure_sim_lines


class CC3DPythonGenerator:

    def __init__(self, xml_generator):

        self.xmlGenerator = xml_generator
        self.simulationDir = self.xmlGenerator.simulationDir
        self.simulationName = self.xmlGenerator.simulationName
        self.xmlFileName = self.xmlGenerator.fileName
        self.mainPythonFileName = os.path.join(str(self.simulationDir), str(self.simulationName) + ".py")
        self.steppablesPythonFileName = os.path.join(str(self.simulationDir),

                                                     str(self.simulationName) + "Steppables.py")

        self.configureSimLines = ''
        self.plotTypeTable = []
        self.pythonPlotsLines = ''
        self.pythonPlotsNames = []
        self.steppableCodeLines = ''
        self.steppableRegistrationLines = ''
        self.generatedSteppableNames = []
        self.generatedVisPlotSteppableNames = []
        self.cellTypeTable = [["Medium", False]]
        self.afMolecules = []
        self.afFormula = 'min(Molecule1,Molecule2)'
        self.cmcCadherins = []
        self.pythonOnlyFlag = False
        self.steppableFrequency = 1

    def set_python_only_flag(self, _flag):

        self.pythonOnlyFlag = _flag

    def generate_configure_sim_fcn(self):

        # note the XML root element generated using C++ xml-to-python converter is called RootElementNameElmnt
        # here it will be CompuCell3DElmnt

        configure_sim_file_name = str(self.xmlFileName + ".py")

        self.configureSimLines = generate_configure_sim_fcn_body(self.xmlGenerator.cc3d.CC3DXMLElement,
                                                                 configure_sim_file_name)

        self.configureSimLines += '\n'

    def generate_main_python_script(self):

        file = open(self.mainPythonFileName, "w")

        print("self.pythonPlotsLines=", self.pythonPlotsLines)

        header = ''

        if self.pythonOnlyFlag:
            self.generate_configure_sim_fcn()

        # note the XML root element generated using C++ xml-to-python converter is called RootElementNameElmnt
        # here it will be CompuCell3DElmnt

        if self.configureSimLines != '':
            header += self.configureSimLines

            header += '''
        
    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)

            '''

        header += '''
from cc3d import CompuCellSetup
        '''

        if self.configureSimLines != '':
            header += '''

configure_simulation()            

            '''

        main_loop_line = '''
CompuCellSetup.run()
'''

        script = header
        script += self.steppableRegistrationLines
        script += main_loop_line
        file.write(script)

        file.close()

    # using only demo steppable
    def generate_steppable_registration_lines(self):

        steppable_module = self.simulationName + "Steppables"
        steppable_class = self.simulationName + "Steppable"
        steppable_frequency = self.steppableFrequency

        if not len(self.generatedSteppableNames) and not len(self.generatedVisPlotSteppableNames):

            self.steppableRegistrationLines += '''

from {steppable_module} import {steppable_class}

CompuCellSetup.register_steppable(steppable={steppable_class}(frequency={steppable_frequency}))

'''.format(steppable_module=steppable_module, steppable_class=steppable_class, steppable_frequency=steppable_frequency)

        else:
            # generating registration lines for user stppables
            for steppable_class in self.generatedSteppableNames:
                self.steppableRegistrationLines += '''


from {steppable_module} import {steppable_class}

CompuCellSetup.register_steppable(steppable={steppable_class}(frequency={steppable_frequency}))

'''.format(steppable_module=steppable_module, steppable_class=steppable_class, steppable_frequency=steppable_frequency)

    def generate_vis_plot_steppables(self):

        if not len(self.pythonPlotsNames):
            return

        self.steppableCodeLines += '''

            

from PlayerPython import *

from math import *            

'''

        for plotNameTuple in self.pythonPlotsNames:

            steppableName = plotNameTuple[0] + 'Steppable'

            if steppableName not in self.generatedVisPlotSteppableNames:
                self.generatedVisPlotSteppableNames.append(steppableName)

            plotType = plotNameTuple[1]

            if plotType == 'ScalarField':

                self.steppableCodeLines += '''



class %s(SteppableBasePy):

''' % (steppableName)

                self.steppableCodeLines += '''

    def __init__(self,_simulator,_frequency=%s):

        SteppableBasePy.__init__(self,_simulator,_frequency)

        self.visField=None

        

    def step(self,mcs):

        clearScalarField(self.dim,self.visField)

        for x in xrange(self.dim.x):

            for y in xrange(self.dim.y):

                for z in xrange(self.dim.z):

                    pt=CompuCell.Point3D(x,y,z)

                    if (not mcs % 20):

                        value=x*y

                        fillScalarValue(self.visField,x,y,z,value) # value assigned to individual pixel

                    else:

                        value=sin(x*y)

                        fillScalarValue(self.visField,x,y,z,value) # value assigned to individual pixel                    

''' % (self.steppableFrequency)



            elif plotType == 'CellLevelScalarField':

                self.steppableCodeLines += '''

                    

class %s(SteppableBasePy):

    def __init__(self,_simulator,_frequency=%s):

        SteppableBasePy.__init__(self,_simulator,_frequency)

        self.visField=None



    def step(self,mcs):

        clearScalarValueCellLevel(self.visField)

        from random import random

        for cell in self.cellList:

            fillScalarValueCellLevel(self.visField,cell,cell.id*random())   # value assigned to every cell , all cell pixels are painted based on this value             

''' % (steppableName, self.steppableFrequency)



            elif plotType == 'VectorField':

                self.steppableCodeLines += '''

                    

class %s(SteppableBasePy):

    def __init__(self,_simulator,_frequency=%s):

        SteppableBasePy.__init__(self,_simulator,_frequency)

        self.visField=None

    

    def step(self,mcs):

        maxLength=0

        clearVectorField(self.dim,self.visField)        

        for x in xrange(0,self.dim.x,5):

            for y in xrange(0,self.dim.y,5):

                for z in xrange(self.dim.z):                     

                    pt=CompuCell.Point3D(x,y,z)                    

                    insertVectorIntoVectorField(self.visField,pt.x, pt.y,pt.z, pt.x, pt.y, pt.z) # vector assigned to individual pixel

''' % (steppableName, self.steppableFrequency)



            elif plotType == 'CellLevelVectorField':

                self.steppableCodeLines += '''

                    

class %s(SteppableBasePy):

    def __init__(self,_simulator,_frequency=%s):

        SteppableBasePy.__init__(self,_simulator,_frequency)

        self.visField=None



    def step(self,mcs):

        clearVectorCellLevelField(self.visField)

        for cell in self.cellList:

            if cell.type==1:

                insertVectorIntoVectorCellLevelField(self.visField,cell, cell.id, cell.id, 0.0)

''' % (steppableName, self.steppableFrequency)

    def generate_constraint_initializer(self):

        if "ConstraintInitializerSteppable" not in self.generatedSteppableNames:
            self.generatedSteppableNames.append("ConstraintInitializerSteppable")

            self.steppableCodeLines += '''

class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self,frequency={steppable_frequency}):
        SteppableBasePy.__init__(self,frequency)

    def start(self):

        for cell in self.cell_list:

            cell.targetVolume = 25
            cell.lambdaVolume = 2.0
        
        '''.format(steppable_frequency=self.steppableFrequency)

    def generate_growth_steppable(self):

        self.generate_constraint_initializer()

        if "GrowthSteppable" not in self.generatedSteppableNames:
            self.generatedSteppableNames.append("GrowthSteppable")

            self.steppableCodeLines += '''
class GrowthSteppable(SteppableBasePy):
    def __init__(self,frequency={steppable_frequency}):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
    
        for cell in self.cell_list:
            cell.targetVolume += 1        

        # # alternatively if you want to make growth a function of chemical concentration uncomment lines below and comment lines above        

        # field = self.field.CHEMICAL_FIELD_NAME
        
        # for cell in self.cell_list:
            # concentrationAtCOM = field[int(cell.xCOM), int(cell.yCOM), int(cell.zCOM)]

            # # you can use here any fcn of concentrationAtCOM
            # cell.targetVolume += 0.01 * concentrationAtCOM       

        ''' .format(steppable_frequency=self.steppableFrequency)

    def generate_mitosis_steppable(self):

        self.generate_growth_steppable()

        if "MitosisSteppable" not in self.generatedSteppableNames:
            self.generatedSteppableNames.append("MitosisSteppable")

            self.steppableCodeLines += '''
class MitosisSteppable(MitosisSteppableBase):
    def __init__(self,frequency={steppable_frequency}):
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

        '''.format(steppable_frequency=self.steppableFrequency)

    def generate_death_steppable(self):

        self.generate_constraint_initializer()

        if "DeathSteppable" not in self.generatedSteppableNames:
            self.generatedSteppableNames.append("DeathSteppable")

            self.steppableCodeLines += '''
class DeathSteppable(SteppableBasePy):
    def __init__(self, frequency={steppable_frequency}):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if mcs == 1000:
            for cell in self.cell_list:
                if cell.type==1:
                    cell.targetVolume=0
                    cell.lambdaVolume=100

        '''.format(steppable_frequency=self.steppableFrequency)

    def generate_steppable_python_script(self):

        file = open(self.steppablesPythonFileName, "w")

        header = '''
from cc3d.core.PySteppables import *

'''
        file.write(header)
        # writing simple demo steppable
        steppable_class = self.simulationName + "Steppable"
        steppable_frequency = self.steppableFrequency
        if self.steppableCodeLines == '':

            class_definition_line = '''class {steppable_class}(SteppableBasePy):'''.format(
                    steppable_class=steppable_class)

            steppable_body = '''

    def __init__(self,frequency={steppable_frequency}):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """

        for cell in self.cell_list:

            print("cell.id=",cell.id)

    def finish(self):
        """
        Finish Function is called after the last MCS
        """

    def on_stop(self):
        # this gets called each time user stops simulation
        return


        '''.format(steppable_frequency=steppable_frequency)

            file.write(class_definition_line)

            file.write(steppable_body)

        else:
            # writing steppables according to user requests
            file.write(self.steppableCodeLines)

        file.close()
