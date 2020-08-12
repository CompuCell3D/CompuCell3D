"""
ideally instead hard-coding snippets we should use XML Schema or RelaxNG formats to describe and help generate CC3DML
And use a single ML generator another one is in
cc3d\twedit5\Plugins\CC3DProject\CC3DXMLGenerator.py

"""
from cc3d.core.XMLUtils import ElementCC3D
import os.path
import functools
import cc3d
from functools import wraps


class GenerateDecorator(object):
    def __init__(self, _moduleType, _moduleName):
        self.moduleType = _moduleType
        self.moduleName = _moduleName

    def __call__(self, _decoratedFn):

        @functools.wraps(_decoratedFn)
        def decorator(*args, **kwds):
            obj = args[0]
            try:
                module_attribute_label = self.moduleName[0]
            except:
                module_attribute_label = ''

            try:
                module_attribute_val = self.moduleName[1]
            except:
                module_attribute_val = ''

            try:
                ir_element = kwds['insert_root_element']
            except LookupError:
                ir_element = None

            # existing root element 

            try:
                r_element = kwds['root_element']
            except LookupError:
                r_element = None

            attr = {}

            if module_attribute_label != '' and module_attribute_val != '':
                attr = {module_attribute_label: module_attribute_val}

            # checking for additional attributes    e.g. Frequency
            try:
                for idx in range(2, len(self.moduleName), 2):
                    module_attribute_label = self.moduleName[idx]
                    module_attribute_val = self.moduleName[idx + 1]
                    attr[module_attribute_label] = module_attribute_val

            except IndexError:
                pass

            # m_element is module element - either steppable of plugin element
            if ir_element is None:
                m_element = ElementCC3D(self.moduleType, attr)

            else:
                ir_element.addComment("newline")
                m_element = ir_element.ElementCC3D(self.moduleType, attr)

            try:

                cell_type_data = kwds['data']

            except LookupError as e:

                cell_type_data = None

            try:

                general_properties_data = kwds['generalPropertiesData']

            except LookupError as e:

                general_properties_data = {}

            gpd = general_properties_data

            module_attribute_label = self.moduleName[0]

            print('CELLTYPE DATA FROM DECORATOR=', cell_type_data)

            obj.cellTypeData = cell_type_data
            obj.mElement = m_element
            obj.gpd = gpd
            _decoratedFn(gpd=gpd, cellTypeData=cell_type_data, *args, **kwds)

            return m_element

        return decorator


class CC3DMLGeneratorBase:

    def __init__(self, simulationDir='', simulationName=''):
        self.element = None

        version = cc3d.__version__
        revision = cc3d.__revision__

        self.cc3d = ElementCC3D("CompuCell3D", {"Version": version, 'Revision': revision})

        self.simulationDir = simulationDir
        self.simulationName = simulationName
        self.fileName = ''

        if self.simulationDir != '' and self.simulationName != '':
            self.fileName = os.path.join(str(self.simulationDir), str(self.simulationName) + ".xml")

    def checkIfSim3D(self, _gpd):

        sim_3d_flag = False

        if _gpd["Dim"][0] > 1 and _gpd["Dim"][1] > 1 and _gpd["Dim"][2] > 1:
            sim_3d_flag = True

        return sim_3d_flag

    # @GenerateDecorator('Potts', ['', ''])
    def getCurrentPottsSection(self, *args, **kwds):

        """

        Returns current Potts section as CC3DXMLElement

        :param args:

        :param kwds:

        :return: {instance of CC3DXMLElement} CC3DXMLElement representing current POtts section

        """

        return kwds['generalPropertiesData']

    @GenerateDecorator('Potts', ['', ''])
    def generatePottsSection(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        gpd = self.gpd

        print('\n\n\n\n gpd=', gpd)

        m_element.addComment("newline")

        m_element.addComment("Basic properties of CPM (GGH) algorithm")
        m_element.ElementCC3D("Dimensions", {"x": gpd["Dim"][0], "y": gpd["Dim"][1], "z": gpd["Dim"][2]})
        m_element.ElementCC3D("Steps", {}, gpd["MCS"])
        m_element.ElementCC3D("Temperature", {}, gpd["MembraneFluctuations"])
        m_element.ElementCC3D("NeighborOrder", {}, gpd["NeighborOrder"])

        if gpd["LatticeType"] != "Square":
            m_element.ElementCC3D("LatticeType", {}, gpd["LatticeType"])

        for dim_name in ['x', 'y', 'z']:

            try:
                if gpd['BoundaryConditions'][dim_name] == 'Periodic':
                    m_element.ElementCC3D('Boundary_' + dim_name, {}, 'Periodic')

            except KeyError:
                m_element.ElementCC3D('Boundary_' + dim_name, {}, 'NoFlux')

    @GenerateDecorator('Metadata', ['', ''])
    def generateMetadataSimulationProperties(self, *args, **kwds):

        cell_type_data = self.cellTypeData

        m_element = self.mElement

        m_element.addComment("newline")

        m_element.addComment("Basic properties simulation")

        m_element.ElementCC3D("NumberOfProcessors", {}, 1)

        m_element.ElementCC3D("DebugOutputFrequency", {}, 10)
        non_parallel_elem = m_element.ElementCC3D("NonParallelModule", {"Name": "Potts"})
        non_parallel_elem.commentOutElement()

    @GenerateDecorator('Metadata', ['', ''])
    def generateMetadataDebugOutputFrequency(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement
        m_element.ElementCC3D("DebugOutputFrequency", {}, 100)

    @GenerateDecorator('Metadata', ['', ''])
    def generateMetadataParallelExecution(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement
        m_element.ElementCC3D("NumberOfProcessors", {}, 2)

    @GenerateDecorator('Metadata', ['', ''])
    def generateMetadataParallelExecutionSingleCPUPotts(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement
        m_element.ElementCC3D("NumberOfProcessors", {}, 2)
        m_element.ElementCC3D("NonParallelModule", {"Name": "Potts"})

    @GenerateDecorator('Plugin', ['Name', 'CellType'])
    def generateCellTypePlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement
        gpd = self.gpd

        m_element.addComment("newline")
        m_element.addComment("Listing all cell types in the simulation")

        for type_id, typeTuple in cell_type_data.items():

            cell_type_dict = {}
            cell_type_dict["TypeName"] = typeTuple[0]
            cell_type_dict["TypeId"] = str(type_id)

            if typeTuple[1]:
                cell_type_dict["Freeze"] = ""

            m_element.ElementCC3D("CellType", cell_type_dict)

    def generateVolumeFlexPlugin(self, *args, **kwds):

        kwds['KeyString'] = 'Volume'

        kwds['KeyType'] = 'Flex'

        return self.volSurHelper(*args, **kwds)

    def generateVolumeLocalFlexPlugin(self, *args, **kwds):

        kwds['KeyString'] = 'Volume'

        kwds['KeyType'] = 'LocalFlex'

        return self.volSurHelper(*args, **kwds)

    def generateSurfaceFlexPlugin(self, *args, **kwds):

        kwds['KeyString'] = 'Surface'
        kwds['KeyType'] = 'Flex'

        return self.volSurHelper(*args, **kwds)

    def generateSurfaceLocalFlexPlugin(self, *args, **kwds):

        kwds['KeyString'] = 'Surface'
        kwds['KeyType'] = 'LocalFlex'

        return self.volSurHelper(*args, **kwds)

    @GenerateDecorator('Plugin', ['Name', 'LengthConstraint'])
    def generateLengthConstraintPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement
        gpd = self.gpd

        try:
            constraint_data_dict = kwds['constraintDataDict']
        except LookupError:
            constraint_data_dict = {}

        m_element.addComment('newline')
        m_element.addComment('Applies elongation constraint to each cell. Users specify target length of '
                             'major axis -TargetLength (in 3D additionally, target length of minor axis - '
                             'MinorTargetLength) and a strength of the constraint -LambdaLength. '
                             'Parameters are specified for each cell type')

        m_element.addComment('IMPORTANT: To prevent cell fragmentation for large elongations '
                             'you need to also use connectivity constraint')

        m_element.addComment('LengthConstraint plugin with no body: <Plugin Name="LengthConstraint"/> '
                             'permits constraint specification for individual cells')

        m_element.addComment("Comment out the constrains for cell types which don't need them")

        sim_3d_flag = self.checkIfSim3D(gpd)
        max_id = max(cell_type_data.keys())

        for id1 in range(0, max_id + 1):
            if cell_type_data[id1][0] == "Medium":
                continue

            target_length = 25.0
            minor_target_length = 5.0
            lambda_val = 2.0

            try:
                data_list = constraint_data_dict[cell_type_data[id1][0]]
            except LookupError:

                data_list = [25, 5.0, 2.0]
            try:

                target_length = data_list[0]
                minor_target_length = data_list[1]
                lambda_val = data_list[2]

            except LookupError:
                pass

            attr = {"CellType": cell_type_data[id1][0], "TargetLength": target_length, "LambdaLength": lambda_val}

            if sim_3d_flag:
                attr["MinorTargetLength"] = minor_target_length

            m_element.ElementCC3D("LengthEnergyParameters", attr)

    def volSurHelper(self, *args, **kwds):

        try:
            ir_element = kwds['insert_root_element']

        except LookupError:
            ir_element = None

        # existing root element

        try:
            r_element = kwds['root_element']
        except LookupError:
            r_element = None

        try:
            cell_type_data = kwds['data']
        except LookupError:
            cell_type_data = None

        try:
            key_string = str(kwds['KeyString'])
        except LookupError:
            key_string = 'Volume'

        try:
            key_type = str(kwds['KeyType'])
        except LookupError:
            key_type = 'LocalFlex'

        try:
            constraint_data_dict = kwds['constraintDataDict']
        except LookupError:
            constraint_data_dict = {}

        # mElement is module element - either steppable of plugin element
        if ir_element is None:

            m_element = ElementCC3D("Plugin", {"Name": key_string})

        else:
            ir_element.addComment("newline")
            m_element = ir_element.ElementCC3D("Plugin", {"Name": key_string})

        if key_type == 'LocalFlex':
            return m_element

        max_id = max(cell_type_data.keys())
        for type_id in range(0, max_id + 1):
            target_val = 50.0
            lambda_val = 2.0

            # Medium
            if type_id == 0:
                continue

            # first see if entry for this type exists
            try:
                data_list = constraint_data_dict[cell_type_data[type_id][0]]
            except LookupError:

                data_list = [50, 2.0]

            try:

                target_val = data_list[0]
                lambda_val = data_list[1]

            except LookupError:
                pass

            attr_dict = {'CellType': cell_type_data[type_id][0], 'Target' + key_string: target_val,

                        'Lambda' + key_string: lambda_val}

            m_element.ElementCC3D(key_string + "EnergyParameters", attr_dict)

        return m_element

    @GenerateDecorator('Plugin', ['Name', 'LengthConstraintLocalFlex'])
    def generateLengthConstraintLocalFlexPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData

        m_element = self.mElement
        gpd = self.gpd

        m_element.addComment("newline")
        m_element.addComment("Applies elongation constraint to each cell. Users specify the length major axis "
                             "-TargetLength and a strength of the constraint -LambdaLength."
                             " Parameters are specified for each cell individually")
        m_element.addComment("IMPORTANT: To prevent cell fragmentation for large elongations "
                             "you need to also use connectivity constraint")

        m_element.addComment("This plugin currently works only in 2D. "
                             "Use the following Python syntax to set/modify length constraint:")

        m_element.addComment("self.lengthConstraintFlexPlugin.setLengthConstraintData(cell,20,30)  "
                             "# cell , lambdaLength, targetLength  ")

    @GenerateDecorator('Plugin', ['Name', 'ExternalPotential'])
    def generateExternalPotentialPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        m_element.addComment("newline")
        m_element.addComment("External force applied to cell. Each cell type has different force.")
        m_element.addComment("For more flexible specification of the constraint (done in Python) "
                             "please use ExternalPotential plugin without specifying per-type parameters")

        m_element.addComment("Algorithm options are: PixelBased, CenterOfMassBased")

        m_element.ElementCC3D("Algorithm", {}, "PixelBased")

        max_id = max(cell_type_data.keys())

        for id1 in range(0, max_id + 1):
            if cell_type_data[id1][0] == "Medium":
                continue

            cell_type_dict = {"CellType": cell_type_data[id1][0], "x": -0.5, "y": 0.0, "z": 0.0}

            m_element.ElementCC3D("ExternalPotentialParameters", cell_type_dict)

    @GenerateDecorator('Plugin', ['Name', 'ExternalPotential'])
    def generateExternalPotentialLocalFlexPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")

        m_element.addComment("External force applied to cell. Each cell has different force and "
                             "force components have to be managed in Python.")

        m_element.addComment("e.g. cell.lambdaVecX=0.5; cell.lambdaVecY=0.1 ; cell.lambdaVecZ=0.3;")

        m_element.ElementCC3D("Algorithm", {}, "PixelBased")

    @GenerateDecorator('Plugin', ['Name', 'CenterOfMass'])
    def generateCenterOfMassPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")

        m_element.addComment("Module tracking center of mass of each cell")

    @GenerateDecorator('Plugin', ['Name', 'NeighborTracker'])
    def generateNeighborTrackerPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")

        m_element.addComment("Module tracking neighboring cells of each cell")

    @GenerateDecorator('Plugin', ['Name', 'MomentOfInertia'])
    def generateMomentOfInertiaPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")

        m_element.addComment("Module tracking moment of inertia of each cell")

    @GenerateDecorator('Plugin', ['Name', 'PixelTracker'])
    def generatePixelTrackerPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")

        m_element.addComment("Module tracking pixels of each cell")

    @GenerateDecorator('Plugin', ['Name', 'BoundaryPixelTracker'])
    def generateBoundaryPixelTrackerPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")

        m_element.addComment("Module tracking boundary pixels of each cell")

        m_element.ElementCC3D("NeighborOrder", {}, 1)

    @GenerateDecorator('Plugin', ['Name', 'CellTypeMonitor'])
    def generateCellTypeMonitorPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")

        m_element.addComment("Module tracking cell types at each lattice site - used mainly by pde solvers")

    @GenerateDecorator('Plugin', ['Name', 'ConnectivityGlobal'])
    def generateConnectivityGlobalPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData

        m_element = self.mElement

        gpd = self.gpd

        m_element.addComment("newline")

        m_element.addComment("Connectivity constraint applied to each cell. Energy penalty specifies how strong "
                             "the constraint is. Penalty is specified for each type ")

        m_element.addComment("This constraint works in 2D and 3D on all type of lattices. "
                             "It might be slowdown your simulation. For faster option - 2D and square"
                             " lattice you may use Connectivity or ConnectivityLocalFlex")

        m_element.addComment("To speed up simulation comment out unnecessary constraints "
                             "for types which don't need the constraint")

        m_element.addComment("By default we will always precheck connectivity BUT in simulations in which "
                             "there is no risk of having unfragmented cell one can add this flag "
                             "to speed up computations")

        m_element.addComment("To turn off precheck uncomment line below")

        precheck_elem = m_element.ElementCC3D("DoNotPrecheckConnectivity")

        precheck_elem.commentOutElement()

        max_id = max(cell_type_data.keys())
        for id1 in range(0, max_id + 1):
            if cell_type_data[id1][0] == "Medium":
                continue

            attr = {"Type": cell_type_data[id1][0]}
            m_element.ElementCC3D("Penalty", attr, 1000000)

    @GenerateDecorator('Plugin', ['Name', 'ConnectivityGlobal'])
    def generateConnectivityGlobalByIdPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")
        m_element.addComment("Connectivity constraint applied to each cell. "
                             "Energy penalty specifies how strong the constraint is. "
                             "Penalty is specified for each cell type individually ")

        m_element.addComment("Use Python scripting to setup penalty (connectivity strength) for each cell")

        m_element.addComment("e.g. self.connectivityGlobalPlugin.setConnectivityStrength(cell,20000000) "
                             "#cell, connectivity strength")

        m_element.addComment("This constraint works in 2D and 3D on all type of lattices. "
                             "It might be slowdown your simulation. For faster option - "
                             "2D and square lattice you may use Connectivity or ConnectivityLocalFlex")

        m_element.addComment(

            "To speed up simulation comment out unnecessary constraints for types which don't need the constraint")

    @GenerateDecorator('Plugin', ['Name', 'Connectivity'])
    def generateConnectivityPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("Connectivity constraint applied to each cell. "
                             "Energy penalty specifies how strong the constraint is. "
                             "Penalty is specified globally for each cell ")

        m_element.addComment("This constraint works in 2D and on square lattice only! "
                             "It also requires that the <NeighborOrder> in the Potts section is 1 or 2!")

        m_element.addComment("For more flexible version of this plugin use ConnectivityLocalFlex "
                             "where constraint penalty is specified for each cell individually "
                             "using Python scripting using the following syntax")

        m_element.addComment("self.connectivityLocalFlexPlugin.setConnectivityStrength(cell,10000000)")

        m_element.ElementCC3D("Penalty", {}, 10000000)

    @GenerateDecorator('Plugin', ['Name', 'Contact'])
    def generateContactPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData

        m_element = self.mElement

        try:
            contact_matrix = kwds['contactMatrix']
        except LookupError:

            contact_matrix = {}

        try:
            n_order = kwds['NeighborOrder']
        except LookupError:
            n_order = 4

        m_element.addComment("Specification of adhesion energies")

        max_id = max(cell_type_data.keys())

        for id1 in range(0, max_id + 1):
            for id2 in range(id1, max_id + 1):
                try:
                    attrDict = {"Type1": cell_type_data[id1][0], "Type2": cell_type_data[id2][0]}
                except LookupError:
                    continue

                try:
                    # first see if energy exists

                    energy = contact_matrix[cell_type_data[id1][0]][cell_type_data[id2][0]][0]

                except LookupError:

                    try:  # try reverse order

                        energy = contact_matrix[cell_type_data[id2][0]][cell_type_data[id1][0]][0]

                    except LookupError:
                        # use default value
                        energy = 10.0

                m_element.ElementCC3D("Energy", attrDict, energy)

        m_element.ElementCC3D("NeighborOrder", {}, n_order)

    @GenerateDecorator('Plugin', ['Name', 'ImplicitMotility'])
    def generateImplicitMotilityPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        try:
            juliano_lambda = kwds['juliano_lambda']
        except KeyError:
            juliano_lambda = 10.0


        max_id = max(cell_type_data.keys())
        for id1 in range(1, max_id + 1):
            try:
                attr_dict = {"Type": cell_type_data[id1][0]}
            except LookupError:
                continue

            m_element.ElementCC3D("Motility", attr_dict, juliano_lambda)

    @GenerateDecorator('Plugin', ['Name', 'ContactInternal'])
    def generateContactInternalPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        try:
            contact_matrix = kwds['contactMatrix']
        except LookupError:
            contact_matrix = {}

        try:
            n_order = kwds['NeighborOrder']
        except LookupError:
            n_order = 4

        m_element.addComment("Specification of internal adhesion energies")

        max_id = max(cell_type_data.keys())

        for id1 in range(1, max_id + 1):
            for id2 in range(id1, max_id + 1):
                try:
                    attr_dict = {"Type1": cell_type_data[id1][0], "Type2": cell_type_data[id2][0]}
                except LookupError:
                    continue

                try:
                    # first see if energy exists
                    energy = contact_matrix[cell_type_data[id1][0]][cell_type_data[id2][0]][0]
                except LookupError:
                    try:
                        # try reverse order
                        energy = contact_matrix[cell_type_data[id2][0]][cell_type_data[id1][0]][0]
                    except LookupError:
                        # use default value
                        energy = 10.0

                m_element.ElementCC3D("Energy", attr_dict, energy)

        m_element.ElementCC3D("NeighborOrder", {}, n_order)

    @GenerateDecorator('Plugin', ['Name', 'Compartment'])
    def generateCompartmentPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        try:
            contact_matrix = kwds['contactMatrix']
        except LookupError:
            contact_matrix = {}

        try:
            internal_contact_matrix = kwds['contactMatrix']
        except LookupError:
            internal_contact_matrix = {}

        try:
            n_order = kwds['NeighborOrder']
        except LookupError:
            n_order = 4

        m_element.addComment("newline")

        m_element.addComment("Specification of adhesion energies in the presence of compartmental cells")

        m_element.addComment(
            "This plugin is deprecated - please consider using Contact and ContactInternal plugins instead")

        m_element.addComment("to specify adhesions bewtween members of same cluster")

        max_id = max(cell_type_data.keys())

        for id1 in range(0, max_id + 1):
            for id2 in range(id1, max_id + 1):
                try:
                    attr_dict = {"Type1": cell_type_data[id1][0], "Type2": cell_type_data[id2][0]}
                except LookupError:
                    continue

                try:
                    # first see if energy exists
                    energy = contact_matrix[cell_type_data[id1][0]][cell_type_data[id2][0]][0]
                except LookupError:

                    try:
                        # try reverse order
                        energy = contact_matrix[cell_type_data[id2][0]][cell_type_data[id1][0]][0]
                    except LookupError:
                        # use default value
                        energy = 10.0

                m_element.ElementCC3D("Energy", attr_dict, energy)

        # energy between members of same clusters

        for id1 in range(1, max_id + 1):
            for id2 in range(id1, max_id + 1):

                try:
                    attr_dict = {"Type1": cell_type_data[id1][0], "Type2": cell_type_data[id2][0]}
                except LookupError:
                    continue

                    # first see if energy exists
                try:
                    internal_energy = internal_contact_matrix[cell_type_data[id1][0]][cell_type_data[id2][0]][0]
                except LookupError:

                    try:  # try reverse order
                        internal_energy = internal_contact_matrix[cell_type_data[id2][0]][cell_type_data[id1][0]][0]
                    except LookupError:
                        # use default value
                        internal_energy = 5.0

                m_element.ElementCC3D("InternalEnergy", attr_dict, internal_energy)

        m_element.ElementCC3D("NeighborOrder", {}, n_order)

    @GenerateDecorator('Plugin', ['Name', 'ContactLocalProduct'])
    def generateContactLocalProductPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        try:
            specificity_matrix = kwds['specificityMatrix']
        except LookupError:
            specificity_matrix = {}

        try:
            n_order = kwds['NeighborOrder']
        except LookupError:
            n_order = 4

        m_element.addComment("newline")
        m_element.addComment(
            "Specification of adhesion energies as a function of cadherin concentration at cell membranes")

        m_element.addComment(
            "Adhesion energy is a function of two cells in ocntact. the functional form is specified by the user")

        m_element.addComment("Please consider using more flexible version of this plugin - AdhesionFlex")

        max_id = max(cell_type_data.keys())

        for id1 in range(0, max_id + 1):
            for id2 in range(id1, max_id + 1):
                attr = {"Type1": cell_type_data[id1][0], "Type2": cell_type_data[id2][0]}

                try:
                    specificity = specificity_matrix[cell_type_data[id1][0]][cell_type_data[id2][0]][0]
                except LookupError:
                    try:
                        # try reverse order
                        specificity = specificity_matrix[cell_type_data[id2][0]][cell_type_data[id1][0]][0]
                    except LookupError:
                        # use default value
                        specificity = -1.0

                m_element.ElementCC3D("ContactSpecificity", attr, specificity)

        m_element.ElementCC3D("ContactFunctionType", {}, "linear")
        m_element.ElementCC3D("EnergyOffset", {}, 0.0)
        m_element.ElementCC3D("NeighborOrder", {}, n_order)

    @GenerateDecorator('Plugin', ['Name', 'FocalPointPlasticity'])
    def generateFocalPointPlasticityPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        try:
            n_order = kwds['NeighborOrder']
        except LookupError:
            n_order = 1

        m_element.addComment("newline")
        m_element.addComment("Specification of focal point junctions")
        m_element.addComment(
            "We separetely specify links between members of same cluster - "
            "InternalParameters and members of different clusters Parameters. "
            "When not using compartmental  cells comment out InternalParameters specification")

        m_element.addComment("To modify FPP links individually for each cell pair uncomment line below")

        localmElement = m_element.ElementCC3D("Local")

        localmElement.commentOutElement()

        m_element.addComment(
            "Note that even though you may manipulate lambdaDistance, "
            "targetDistance and maxDistance using Python you still need to set activation energy from XML level")

        m_element.addComment("See CC3D manual for details on FPP plugin ")

        max_id = max(cell_type_data.keys())

        for id1 in range(1, max_id + 1):
            for id2 in range(id1, max_id + 1):
                m_element.addComment("newline")
                attr = {"Type1": cell_type_data[id1][0], "Type2": cell_type_data[id2][0]}

                param_element = m_element.ElementCC3D("Parameters", attr)
                param_element.ElementCC3D("Lambda", {}, 10)
                param_element.ElementCC3D("ActivationEnergy", {}, -50)
                param_element.ElementCC3D("TargetDistance", {}, 7)
                param_element.ElementCC3D("MaxDistance", {}, 20)
                param_element.ElementCC3D("MaxNumberOfJunctions", {"NeighborOrder": 1}, 1)

        for id1 in range(1, max_id + 1):
            for id2 in range(id1, max_id + 1):
                m_element.addComment("newline")

                attr = {"Type1": cell_type_data[id1][0], "Type2": cell_type_data[id2][0]}

                param_element = m_element.ElementCC3D("InternalParameters", attr)
                param_element.ElementCC3D("Lambda", {}, 10)
                param_element.ElementCC3D("ActivationEnergy", {}, -50)
                param_element.ElementCC3D("TargetDistance", {}, 7)
                param_element.ElementCC3D("MaxDistance", {}, 20)
                param_element.ElementCC3D("MaxNumberOfJunctions", {"NeighborOrder": 1}, 1)

        m_element.addComment("newline")

        m_element.ElementCC3D("NeighborOrder", {}, n_order)

    @GenerateDecorator('Plugin', ['Name', 'ElasticityTracker'])
    def generateElasticityTrackerPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        m_element.addComment("newline")
        m_element.addComment(
            "Elastic constraints between Center of mass of cells."
            " Need to be accompanied by ElasticityTracker plugin to work. "
            "Only cells in contact at MCS=0 will be affected by the constraint")

        m_element.addComment(
            "ElasticityTracker keeps track of cell neighbors which are "
            "participating in the elasticity constraint calculations")

        m_element.addComment("Comment out cell types which should be unaffected by the constraint")

        max_id = max(cell_type_data.keys())

        for id1 in range(0, max_id + 1):
            if cell_type_data[id1][0] == "Medium":
                continue

            m_element.ElementCC3D("IncludeType", {}, cell_type_data[id1][0])

    @GenerateDecorator('Plugin', ['Name', 'Elasticity'])
    def generateElasticityPlugin(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")
        m_element.addComment("This plugin calculates elastic constraints between cells Center of Mass")
        m_element.addComment("To enable specification of elastic links individually for each link uncomment line below ")

        local_elem = m_element.ElementCC3D("Local")
        local_elem.commentOutElement()

        m_element.addComment("See CC3D manual for details")
        m_element.ElementCC3D("LambdaElasticity", {}, 200)
        m_element.ElementCC3D("TargetLengthElasticity", {}, 6)

    @GenerateDecorator('Plugin', ['Name', 'AdhesionFlex'])
    def generateAdhesionFlexPlugin(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        try:
            af_data = kwds['afData']
        except LookupError:
            af_data = {}

        try:
            formula = kwds['formula']
        except LookupError:

            formula = ''

        m_element.addComment("newline")

        m_element.addComment(
            "Specification of adhesion energies as a function of cadherin concentration at cell membranes")

        m_element.addComment(
            "Adhesion energy is a function of two cells in ocntact. the functional form is specified by the user")

        # writing AdhesionMolecule elements

        for idx, props in af_data.items():
            attr_dict = {"Molecule": props}
            m_element.ElementCC3D("AdhesionMolecule", attr_dict)

        # writing AdhesionMoleculeDensity elements
        for type_id, props in cell_type_data.items():

            for idx, afprops in af_data.items():
                attr_dict = {"CellType": props[0], "Molecule": afprops, "Density": 1.1}
                m_element.ElementCC3D("AdhesionMoleculeDensity", attr_dict)

        # writing binding formula

        bf_element = m_element.ElementCC3D("BindingFormula", {'Name': 'Binary'})

        bf_element.ElementCC3D("Formula", {}, formula)

        var_element = bf_element.ElementCC3D("Variables")

        adh_matrix_element = var_element.ElementCC3D("AdhesionInteractionMatrix")

        repetition_dict = {}

        for idx1, afprops1 in af_data.items():
            for idx2, afprops2 in af_data.items():
                if afprops2 + '_' + afprops1 in list(repetition_dict.keys()):  # to avoid duplicate entries
                    continue
                else:
                    repetition_dict[afprops1 + '_' + afprops2] = 0
                attr_dict = {"Molecule1": afprops1, "Molecule2": afprops2}
                adh_matrix_element.ElementCC3D("BindingParameter", attr_dict, 0.5)

        m_element.ElementCC3D("NeighborOrder", {}, 4)

    @GenerateDecorator('Plugin', ['Name', 'Chemotaxis'])
    def generateChemotaxisPlugin(self, *args, **kwds):

        m_element = self.mElement

        try:
            chemotaxis_data = kwds['chemotaxisData']
        except LookupError:

            chemotaxis_data = {}

        try:
            pde_field_data = kwds['pdeFieldData']
        except LookupError:
            pde_field_data = {}

        m_element.addComment("newline")
        m_element.addComment('You may repeat ChemicalField element for each chemical field declared in the PDE solvers')
        m_element.addComment("Specification of chemotaxis properties of select cell types.")

        for chem_field_name, chem_dict_list in chemotaxis_data.items():

            chem_field_element = m_element.ElementCC3D("ChemicalField", {"Name": chem_field_name})

            for chem_dict in chem_dict_list:
                lambda_ = chem_dict["Lambda"]
                chemotaxTowards = chem_dict["ChemotaxTowards"]
                satCoef = chem_dict["SatCoef"]
                chemotaxisType = chem_dict["ChemotaxisType"]

                attribute_dict = {}
                attribute_dict["Type"] = chem_dict["CellType"]
                attribute_dict["Lambda"] = chem_dict["Lambda"]

                if chem_dict["ChemotaxTowards"] != '':
                    attribute_dict["ChemotactTowards"] = chem_dict["ChemotaxTowards"]

                if chem_dict["ChemotaxisType"] == 'saturation':
                    attribute_dict["SaturationCoef"] = chem_dict["SatCoef"]

                elif chem_dict["ChemotaxisType"] == 'saturation linear':
                    attribute_dict["SaturationLinearCoef"] = chem_dict["SatCoef"]

                chem_field_element.ElementCC3D("ChemotaxisByType", attribute_dict)

    @GenerateDecorator('Plugin', ['Name', 'Secretion'])
    def generateSecretionPlugin(self, *args, **kwds):

        m_element = self.mElement

        try:
            secretion_data = kwds['secretionData']
        except LookupError:

            secretion_data = {}

        m_element.addComment("newline")
        m_element.addComment("Specification of secretion properties of select cell types.")
        m_element.addComment('You may repeat Field element for each chemical field declared in the PDE solvers')
        m_element.addComment("Specification of secretion properties of individual cells can be done in Python")

        for chem_field_name, secr_dict_list in secretion_data.items():

            secr_field_element = m_element.ElementCC3D("Field", {"Name": chem_field_name})
            for secr_dict in secr_dict_list:

                rate = secr_dict["Rate"]
                attribute_dict = {}

                attribute_dict["Type"] = secr_dict["CellType"]
                if secr_dict["SecretionType"] == 'uniform':
                    secr_field_element.ElementCC3D("Secretion", attribute_dict, rate)

                elif secr_dict["SecretionType"] == 'on contact':
                    attribute_dict["SecreteOnContactWith"] = secr_dict["OnContactWith"]
                    secr_field_element.ElementCC3D("SecretionOnContact", attribute_dict, rate)

                elif secr_dict["SecretionType"] == 'constant concentration':
                    secr_field_element.ElementCC3D("ConstantConcentration", attribute_dict, rate)

    @GenerateDecorator('Steppable', ['Type', 'UniformInitializer'])
    def generateUniformInitializerSteppable(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement

        gpd = self.gpd

        m_element.addComment("newline")
        m_element.addComment("Initial layout of cells in the form of rectangular slab")
        region = m_element.ElementCC3D("Region")

        x_min = int(gpd["Dim"][0] * 0.2)
        x_max = int(gpd["Dim"][0] * 0.8)

        if x_max == 0:
            x_max += 1

        y_min = int(gpd["Dim"][1] * 0.2)
        y_max = int(gpd["Dim"][1] * 0.8)

        if y_max == 0:
            y_max += 1

        z_min = int(gpd["Dim"][2] * 0.2)
        z_max = int(gpd["Dim"][2] * 0.8)

        if z_max == 0:
            z_max += 1

        region.ElementCC3D("BoxMin", {"x": x_min, "y": y_min, "z": z_min})

        region.ElementCC3D("BoxMax", {"x": x_max, "y": y_max, "z": z_max})

        region.ElementCC3D("Gap", {}, 0)

        region.ElementCC3D("Width", {}, 7)

        types_string = ""

        max_id = max(cell_type_data.keys())

        for id1 in range(0, max_id + 1):
            if cell_type_data[id1][0] == "Medium":
                continue

            types_string += cell_type_data[id1][0]

            if id1 < max_id:
                types_string += ","

        region.ElementCC3D("Types", {}, types_string)

    @GenerateDecorator('Steppable', ['Type', 'BlobInitializer'])
    def generateBlobInitializerSteppable(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement
        gpd = self.gpd

        m_element.addComment("newline")
        m_element.addComment("Initial layout of cells in the form of spherical (circular in 2D) blob")

        region = m_element.ElementCC3D("Region")

        x_center = int(gpd["Dim"][0] / 2)
        y_center = int(gpd["Dim"][1] / 2)
        z_center = int(gpd["Dim"][2] / 2)

        max_dim = max([x_center, y_center, z_center])

        region.ElementCC3D("Center", {"x": x_center, "y": y_center, "z": z_center})
        region.ElementCC3D("Radius", {}, int(max_dim / 2.5))
        region.ElementCC3D("Gap", {}, 0)
        region.ElementCC3D("Width", {}, 7)

        types_string = ""

        max_id = max(cell_type_data.keys())

        for id1 in range(0, max_id + 1):
            if cell_type_data[id1][0] == "Medium":
                continue

            types_string += cell_type_data[id1][0]

            if id1 < max_id:
                types_string += ","

        region.ElementCC3D("Types", {}, types_string)

    @GenerateDecorator('Steppable', ['Type', 'PIFInitializer'])
    def generatePIFInitializerSteppable(self, *args, **kwds):

        m_element = self.mElement
        gpd = self.gpd

        m_element.addComment("newline")
        m_element.addComment("Initial layout of cells using PIFF file. Piff files can be generated using PIFGEnerator")

        try:
            if gpd["Initializer"][0] == "piff":
                m_element.ElementCC3D("PIFName", {}, gpd["Initializer"][1])

            else:
                m_element.ElementCC3D("PIFName", {}, "PLEASE_PUT_PROPER_FILE_NAME_HERE")
        except:

            m_element.ElementCC3D("PIFName", {}, "PLEASE_PUT_PROPER_FILE_NAME_HERE")

    @GenerateDecorator('Steppable', ['Type', 'PIFDumper', 'Frequency', '100'])
    def generatePIFDumperSteppable(self, *args, **kwds):

        m_element = self.mElement
        gpd = self.gpd

        m_element.addComment("newline")
        m_element.addComment("Periodically stores cell layout configuration in a piff format")
        m_element.ElementCC3D("PIFName", {}, gpd['SimulationName'])
        m_element.ElementCC3D("PIFFileExtension", {}, "piff")

    @GenerateDecorator('Steppable', ['Type', 'BoxWatcher'])
    def generateBoxWatcherSteppable(self, *args, **kwds):

        m_element = self.mElement

        m_element.addComment("newline")
        m_element.addComment(
            "Module tracing boundaries of the minimal box enclosing all the cells. "
            "May speed up calculations. May have no effect for parallel version")

        m_element.ElementCC3D("XMargin", {}, 7)

        m_element.ElementCC3D("YMargin", {}, 7)

        m_element.ElementCC3D("ZMargin", {}, 7)

    @GenerateDecorator('Steppable', ['Type', 'DiffusionSolverFE'])
    def generateDiffusionSolverFE(self, *args, **kwds):

        cell_type_data = self.cellTypeData
        m_element = self.mElement
        gpd = self.gpd

        try:
            pde_field_data = kwds['pdeFieldData']
        except LookupError as e:
            pde_field_data = {}

        sim_3d_flag = self.checkIfSim3D(gpd)

        m_element.addComment("newline")
        m_element.addComment("Specification of PDE solvers")

        for field_name, solver in pde_field_data.items():

            if solver == 'DiffusionSolverFE':

                diff_field_elem = m_element.ElementCC3D("DiffusionField", {"Name": field_name})

                diff_data = diff_field_elem.ElementCC3D("DiffusionData")
                diff_data.ElementCC3D("FieldName", {}, field_name)
                diff_data.ElementCC3D("GlobalDiffusionConstant", {}, 0.1)
                diff_data.ElementCC3D("GlobalDecayConstant", {}, 0.00001)
                diff_data.addComment("Additional options are:")

                conc_eqn_elem = diff_data.ElementCC3D("InitialConcentrationExpression", {}, "x*y")
                conc_eqn_elem.commentOutElement()

                conc_field_name_elem = diff_data.ElementCC3D("ConcentrationFileName", {},
                     "INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")

                conc_field_name_elem.commentOutElement()

                max_id = max(cell_type_data.keys())

                for id1 in range(0, max_id + 1):
                    if cell_type_data[id1][0] == "Medium":
                        continue

                    diff_data.ElementCC3D('DiffusionCoefficient', {'CellType': cell_type_data[id1][0]}, 0.1)

                for id1 in range(0, max_id + 1):
                    if cell_type_data[id1][0] == "Medium":
                        continue

                    diff_data.ElementCC3D('DecayCoefficient', {'CellType': cell_type_data[id1][0]}, 0.0001)

                secr_data = diff_field_elem.ElementCC3D("SecretionData")
                secr_data.addComment(
                    'When secretion is defined inside DissufionSolverFE all secretion constants are scaled '
                    'automaticaly to account for the extra calls to the diffusion step '
                    'when handling large diffusion constants')

                secr_data.addComment('newline')
                secr_data.addComment('Uniform secretion Definition')

                max_id = max(cell_type_data.keys())
                for id1 in range(0, max_id + 1):
                    if cell_type_data[id1][0] == "Medium":
                        continue

                    secr_data.ElementCC3D("Secretion", {"Type": cell_type_data[id1][0]}, 0.1)

                secrete_on_contact_with = ''

                example_type = ''
                for id1 in range(0, max_id + 1):
                    if cell_type_data[id1][0] == "Medium":
                        continue

                    if secrete_on_contact_with != '':
                        secrete_on_contact_with += ','

                    secrete_on_contact_with += cell_type_data[id1][0]
                    example_type = cell_type_data[id1][0]

                secr_on_contact_elem = secr_data.ElementCC3D("SecretionOnContact", {'Type': example_type,
                    "SecreteOnContactWith": secrete_on_contact_with}, 0.2)

                secr_on_contact_elem.commentOutElement()

                const_conc_elem = secr_data.ElementCC3D("ConstantConcentration", {"Type": example_type}, 0.1)

                const_conc_elem.commentOutElement()

                # Boundary Conditions
                bc_data = diff_field_elem.ElementCC3D("BoundaryConditions")

                plane_x_elem = bc_data.ElementCC3D("Plane", {'Axis': 'X'})
                plane_x_elem.ElementCC3D("ConstantValue", {'PlanePosition': 'Min', 'Value': 10.0})
                plane_x_elem.ElementCC3D("ConstantValue", {'PlanePosition': 'Max', 'Value': 5.0})
                plane_x_elem.addComment("Other options are (examples):")

                periodic_x_elem = plane_x_elem.ElementCC3D("Periodic")

                periodic_x_elem.commentOutElement()

                cd_elem = plane_x_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})

                cd_elem.commentOutElement()

                plane_y_elem = bc_data.ElementCC3D("Plane", {'Axis': 'Y'})
                plane_y_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                plane_y_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Max', 'Value': 5.0})
                plane_y_elem.addComment("Other options are (examples):")

                periodic_y_elem = plane_y_elem.ElementCC3D("Periodic")
                periodic_y_elem.commentOutElement()

                cv_elem = plane_y_elem.ElementCC3D('ConstantValue', {'PlanePosition': 'Min', 'Value': 10.0})
                cv_elem.commentOutElement()

                if sim_3d_flag:
                    plane_z_elem = bc_data.ElementCC3D("Plane", {'Axis': 'Z'})
                    plane_z_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                    plane_z_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Max', 'Value': 5.0})
                    plane_z_elem.addComment("Other options are (examples):")

                    periodic_z_elem = plane_z_elem.ElementCC3D("Periodic")
                    periodic_z_elem.commentOutElement()

                    cvz_elem = plane_z_elem.ElementCC3D('ConstantValue', {'PlanePosition': 'Min', 'Value': 10.0})

                    cvz_elem.commentOutElement()

    @GenerateDecorator('Steppable', ['Type', 'FlexibleDiffusionSolverFE'])
    def generateFlexibleDiffusionSolverFE(self, *args, **kwds):

        m_element = self.mElement
        gpd = self.gpd

        try:
            pde_field_data = kwds['pdeFieldData']
        except LookupError:
            pde_field_data = {}

        sim_3d_flag = self.checkIfSim3D(gpd)

        m_element.addComment("newline")
        m_element.addComment("Specification of PDE solvers")

        for field_name, solver in pde_field_data.items():

            if solver == 'FlexibleDiffusionSolverFE':

                diff_field_elem = m_element.ElementCC3D("DiffusionField", {"Name": field_name})
                diff_data = diff_field_elem.ElementCC3D("DiffusionData")
                diff_data.ElementCC3D("FieldName", {}, field_name)
                diff_data.ElementCC3D("DiffusionConstant", {}, 0.1)
                diff_data.ElementCC3D("DecayConstant", {}, 0.00001)
                diff_data.addComment("Additional options are:")

                donotdiffuse_elem = diff_data.ElementCC3D("DoNotDiffuseTo", {}, "LIST YOUR CELL TYPES HERE")

                donotdiffuse_elem.commentOutElement()

                donotdecayin_elem = diff_data.ElementCC3D("DoNotDecayIn", {}, "LIST YOUR CELL TYPES HERE")

                donotdecayin_elem.commentOutElement()

                conc_eqn_elem = diff_data.ElementCC3D("InitialConcentrationExpression", {}, "x*y")
                conc_eqn_elem.commentOutElement()

                conc_field_name_elem = diff_data.ElementCC3D("ConcentrationFileName", {},
                    "INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")

                conc_field_name_elem.commentOutElement()

                diff_data.addComment(
                    "To run solver for large diffusion constants you typically call solver multiple times - "
                    "ExtraTimesPerMCS to specify additional calls to the solver in each MCS ")

                diff_data.addComment(
                    "IMPORTANT: make sure not to mix this setting with the PDESolverCaller module! "
                    "See manual for more information")

                extra_times_per_mcs_elem = diff_data.ElementCC3D("ExtraTimesPerMCS", {}, 0)

                extra_times_per_mcs_elem.commentOutElement()

                delta_x_elem = diff_data.ElementCC3D("DeltaX", {}, 1.0)
                delta_x_elem.commentOutElement()

                delta_t_elem = diff_data.ElementCC3D("DeltaT", {}, 1.0)
                delta_t_elem.commentOutElement()

                # Boiundary Conditions

                bc_data = diff_field_elem.ElementCC3D("BoundaryConditions")

                plane_x_elem = bc_data.ElementCC3D("Plane", {'Axis': 'X'})

                plane_x_elem.ElementCC3D("ConstantValue", {'PlanePosition': 'Min', 'Value': 10.0})
                plane_x_elem.ElementCC3D("ConstantValue", {'PlanePosition': 'Max', 'Value': 5.0})
                plane_x_elem.addComment("Other options are (examples):")

                periodic_x_elem = plane_x_elem.ElementCC3D("Periodic")
                periodic_x_elem.commentOutElement()

                cd_elem = plane_x_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                cd_elem.commentOutElement()

                plane_y_elem = bc_data.ElementCC3D("Plane", {'Axis': 'Y'})
                plane_y_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                plane_y_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Max', 'Value': 5.0})
                plane_y_elem.addComment("Other options are (examples):")

                periodic_y_elem = plane_y_elem.ElementCC3D("Periodic")
                periodic_y_elem.commentOutElement()

                cv_elem = plane_y_elem.ElementCC3D('ConstantValue', {'PlanePosition': 'Min', 'Value': 10.0})
                cv_elem.commentOutElement()

                if sim_3d_flag:
                    plane_z_elem = bc_data.ElementCC3D("Plane", {'Axis': 'Z'})
                    plane_z_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                    plane_z_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Max', 'Value': 5.0})
                    plane_z_elem.addComment("Other options are (examples):")

                    periodic_z_elem = plane_z_elem.ElementCC3D("Periodic")
                    periodic_z_elem.commentOutElement()

                    cvz_elem = plane_z_elem.ElementCC3D('ConstantValue', {'PlanePosition': 'Min', 'Value': 10.0})
                    cvz_elem.commentOutElement()

    @GenerateDecorator('Steppable', ['Type', 'FastDiffusionSolver2DFE'])
    def generateFastDiffusionSolver2DFE(self, *args, **kwds):

        m_element = self.mElement

        try:
            pde_field_data = kwds['pdeFieldData']
        except LookupError:
            pde_field_data = {}

        m_element.addComment("newline")

        m_element.addComment("Specification of PDE solvers")
        for field_name, solver in pde_field_data.items():

            if solver == 'FastDiffusionSolver2DFE':
                diff_field_elem = m_element.ElementCC3D("DiffusionField", {"Name": field_name})

                diff_data = diff_field_elem.ElementCC3D("DiffusionData")
                diff_data.ElementCC3D("FieldName", {}, field_name)
                diff_data.ElementCC3D("DiffusionConstant", {}, 0.1)
                diff_data.ElementCC3D("DecayConstant", {}, 0.00001)
                diff_data.addComment("Additional options are:")

                donotdiffuse_elem = diff_data.ElementCC3D("DoNotDiffuseTo", {}, "LIST YOUR CELL TYPES HERE")
                donotdiffuse_elem.commentOutElement()

                donotdecayin_elem = diff_data.ElementCC3D("DoNotDecayIn", {}, "LIST YOUR CELL TYPES HERE")
                donotdecayin_elem.commentOutElement()

                conc_eqn_elem = diff_data.ElementCC3D("InitialConcentrationExpression", {}, "x*y")
                conc_eqn_elem.commentOutElement()

                conc_field_name_elem = diff_data.ElementCC3D("ConcentrationFileName", {},
                    "INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")

                conc_field_name_elem.commentOutElement()

                diff_data.addComment(
                    "To run solver for large diffusion constants you typically call solver multiple times - "
                    "ExtraTimesPerMCS to specify additional calls to the solver in each MCS ")

                diff_data.addComment(
                    "IMPORTANT: make sure not to mix this setting with the PDESolverCaller module! "
                    "See manual for more information")

                extra_times_per_mcs_elem = diff_data.ElementCC3D("ExtraTimesPerMCS", {}, 0)
                extra_times_per_mcs_elem.commentOutElement()

                delta_x_elem = diff_data.ElementCC3D("DeltaX", {}, 1.0)
                delta_x_elem.commentOutElement()

                delta_t_elem = diff_data.ElementCC3D("DeltaT", {}, 1.0)
                delta_t_elem.commentOutElement()

                # Boundary Conditions
                bc_data = diff_field_elem.ElementCC3D("BoundaryConditions")

                plane_x_elem = bc_data.ElementCC3D("Plane", {'Axis': 'X'})
                plane_x_elem.ElementCC3D("ConstantValue", {'PlanePosition': 'Min', 'Value': 10.0})

                plane_x_elem.ElementCC3D("ConstantValue", {'PlanePosition': 'Max', 'Value': 5.0})

                plane_x_elem.addComment("Other options are (examples):")

                periodic_x_elem = plane_x_elem.ElementCC3D("Periodic")

                periodic_x_elem.commentOutElement()

                cd_elem = plane_x_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                cd_elem.commentOutElement()

                plane_y_elem = bc_data.ElementCC3D("Plane", {'Axis': 'Y'})

                plane_y_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                plane_y_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Max', 'Value': 5.0})
                plane_y_elem.addComment("Other options are (examples):")

                periodic_y_elem = plane_y_elem.ElementCC3D("Periodic")

                periodic_y_elem.commentOutElement()

                cv_elem = plane_y_elem.ElementCC3D('ConstantValue', {'PlanePosition': 'Min', 'Value': 10.0})
                cv_elem.commentOutElement()

    @GenerateDecorator('Steppable', ['Type', 'KernelDiffusionSolver'])
    def generateKernelDiffusionSolver(self, *args, **kwds):

        m_element = self.mElement

        try:
            pde_field_data = kwds['pdeFieldData']
        except LookupError:
            pde_field_data = {}

        m_element.addComment("newline")
        m_element.addComment("Specification of PDE solvers")

        for field_name, solver in pde_field_data.items():

            if solver == 'KernelDiffusionSolver':
                diff_field_elem = m_element.ElementCC3D("DiffusionField", {"Name": field_name})
                diff_field_elem.ElementCC3D("Kernel", {}, "4")

                diff_data = diff_field_elem.ElementCC3D("DiffusionData")
                diff_data.ElementCC3D("FieldName", {}, field_name)
                diff_data.ElementCC3D("DiffusionConstant", {}, 0.1)
                diff_data.ElementCC3D("DecayConstant", {}, 0.00001)

                diff_data.addComment("Additional options are:")

                donotdiffuse_elem = diff_data.ElementCC3D("DoNotDiffuseTo", {}, "LIST YOUR CELL TYPES HERE")

                donotdiffuse_elem.commentOutElement()

                donotdecayin_elem = diff_data.ElementCC3D("DoNotDecayIn", {}, "LIST YOUR CELL TYPES HERE")

                donotdecayin_elem.commentOutElement()

                conc_eqn_elem = diff_data.ElementCC3D("InitialConcentrationExpression", {}, "x*y")

                conc_eqn_elem.commentOutElement()
                conc_field_name_elem = diff_data.ElementCC3D("ConcentrationFileName", {},
                    "INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")

                conc_field_name_elem.commentOutElement()

                diff_data.ElementCC3D("DeltaX", {}, 1.0)

                diff_data.ElementCC3D("DeltaT", {}, 1.0)

    # @GenerateDecorator('Steppable',['Type','SteadyStateDiffusionSolver'])
    def generateSteadyStateDiffusionSolver(self, *args, **kwds):

        try:
            ir_element = kwds['insert_root_element']
        except LookupError:
            ir_element = None

        # existing root element
        try:

            r_element = kwds['root_element']

        except LookupError:
            r_element = None

        try:
            general_properties_data = kwds['generalPropertiesData']
        except LookupError:

            general_properties_data = {}

        gpd = general_properties_data

        sim_3d_flag = self.checkIfSim3D(gpd)

        solver_name = 'SteadyStateDiffusionSolver'

        if not sim_3d_flag:
            solver_name += '2D'

        # mElement is module element - either steppable of plugin element

        if ir_element is None:
            m_element = ElementCC3D("Steppable", {"Type": solver_name})

        else:
            ir_element.addComment("newline")
            m_element = ir_element.ElementCC3D("Steppable", {"Type": solver_name})

        try:
            pde_field_data = kwds['pdeFieldData']
        except LookupError:
            pde_field_data = {}

        try:
            cell_type_data = kwds['data']
        except LookupError:
            cell_type_data = None

        m_element.addComment("newline")

        m_element.addComment("Specification of PDE solvers")

        for fieldName, solver in pde_field_data.items():
            if solver == 'SteadyStateDiffusionSolver':

                diff_field_elem = m_element.ElementCC3D("DiffusionField", {"Name": fieldName})

                diff_data = diff_field_elem.ElementCC3D("DiffusionData")
                diff_data.ElementCC3D("FieldName", {}, fieldName)
                diff_data.ElementCC3D("DiffusionConstant", {}, 1.0)
                diff_data.ElementCC3D("DecayConstant", {}, 0.00001)

                conc_eqn_elem = diff_data.ElementCC3D("InitialConcentrationExpression", {}, "x*y")
                conc_eqn_elem.commentOutElement()

                conc_field_name_elem = diff_data.ElementCC3D("ConcentrationFileName", {},
                    "INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")

                conc_field_name_elem.commentOutElement()

                secr_data = diff_field_elem.ElementCC3D("SecretionData")

                secr_data.addComment(
                    'Secretion has to be defined inside SteadyStateDissufion solver - '
                    'Secretion Plugin doe s not work with this solver.')

                secr_data.addComment('newline')
                secr_data.addComment('Uniform secretion Definition')
                secr_data.ElementCC3D("Secretion", {"Type": 'CELL TYPE 1'}, 0.1)
                secr_data.ElementCC3D("Secretion", {"Type": 'CELL TYPE 2'}, 0.2)

                # Boiundary Conditions

                bc_data = diff_field_elem.ElementCC3D("BoundaryConditions")

                plane_x_elem = bc_data.ElementCC3D("Plane", {'Axis': 'X'})
                plane_x_elem.ElementCC3D("ConstantValue", {'PlanePosition': 'Min', 'Value': 10.0})
                plane_x_elem.ElementCC3D("ConstantValue", {'PlanePosition': 'Max', 'Value': 5.0})
                plane_x_elem.addComment("Other options are (examples):")

                periodic_x_elem = plane_x_elem.ElementCC3D("Periodic")
                periodic_x_elem.commentOutElement()

                cd_elem = plane_x_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})

                cd_elem.commentOutElement()

                plane_y_elem = bc_data.ElementCC3D("Plane", {'Axis': 'Y'})

                plane_y_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                plane_y_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Max', 'Value': 5.0})
                plane_y_elem.addComment("Other options are (examples):")

                periodic_y_elem = plane_y_elem.ElementCC3D("Periodic")
                periodic_y_elem.commentOutElement()

                cv_elem = plane_y_elem.ElementCC3D('ConstantValue', {'PlanePosition': 'Min', 'Value': 10.0})
                cv_elem.commentOutElement()

                if sim_3d_flag:
                    plane_z_elem = bc_data.ElementCC3D("Plane", {'Axis': 'Z'})
                    plane_z_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Min', 'Value': 10.0})
                    plane_z_elem.ElementCC3D('ConstantDerivative', {'PlanePosition': 'Max', 'Value': 5.0})
                    plane_z_elem.addComment("Other options are (examples):")

                    periodic_z_elem = plane_z_elem.ElementCC3D("Periodic")

                    periodic_z_elem.commentOutElement()

                    cvz_elem = plane_z_elem.ElementCC3D('ConstantValue', {'PlanePosition': 'Min', 'Value': 10.0})
                    cvz_elem.commentOutElement()

        return m_element

    def saveCC3DXML(self, _fileName):

        print("SAVING XML = ", _fileName)

        xml_file = open(_fileName, 'w')
        xml_file.write('%s' % self.cc3d.CC3DXMLElement.getCC3DXMLElementString())
        xml_file.close()

