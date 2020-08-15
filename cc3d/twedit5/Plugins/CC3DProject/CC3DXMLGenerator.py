from cc3d.core.XMLUtils import ElementCC3D
import cc3d
import os.path


class CC3DXMLGenerator:

    def __init__(self, simulation_dir, simulation_name):
        self.simulationDir = simulation_dir
        self.simulationName = simulation_name

        self.fileName = os.path.join(str(self.simulationDir), str(self.simulationName) + ".xml")

        self.generalPropertiesDict = {}

        self.cellTypeTable = [["Medium", False]]

        self.cc3d = ElementCC3D("CompuCell3D", {"version": cc3d.__version__})
        self.afMolecules = []
        self.afFormula = 'min(Molecule1,Molecule2)'
        self.cmcCadherins = []
        self.chemFieldsTable = []
        self.secretionTable = {}
        self.chemotaxisTable = {}

    def setGeneralPropertiesDict(self, _dict):
        """

        :param _dict:
        :return:
        """

        self.generalPropertiesDict = _dict

    def setChemFieldsTable(self, _table):
        """

        :param _table:
        :return:
        """

        self.chemFieldsTable = _table

    def setSecretionTable(self, _table):
        """

        :param _table:
        :return:
        """

        self.secretionTable = _table

    def setChemotaxisTable(self, _table):
        """

        :param _table:
        :return:
        """

        self.chemotaxisTable = _table

    def setCMCTable(self, _table):
        """

        :param _table:
        :return:
        """

        self.cmcCadherins = _table

    def setAFFormula(self, _formula):
        """

        :param _formula:
        :return:
        """
        self.afFormula = _formula

    def setAFTable(self, _table):
        """

        :param _table:
        :return:
        """
        self.afMolecules = _table

    def setCellTypeTable(self, _table):

        self.cellTypeTable = _table

        # generate typeId to typeTuple lookup dictionary

        self.idToTypeTupleDict = {}

        type_counter = 0

        for typeTupple in self.cellTypeTable:

            self.idToTypeTupleDict[type_counter] = typeTupple

            type_counter += 1

    def checkIfSim3D(self):

        sim_3d_flag = False

        gpd = self.generalPropertiesDict

        if gpd["Dim"][0] > 1 and gpd["Dim"][1] > 1 and gpd["Dim"][2] > 1:

            sim_3d_flag = True

        return sim_3d_flag

    def generatePottsSection(self):
        """
        generates Potts Section
        :return:
        """

        self.cc3d.addComment("newline")

        self.cc3d.addComment("Basic properties of CPM (GGH) algorithm")

        potts = self.cc3d.ElementCC3D("Potts")
        gpd = self.generalPropertiesDict
        potts.ElementCC3D("Dimensions", {"x": gpd["Dim"][0], "y": gpd["Dim"][1], "z": gpd["Dim"][2]})
        potts.ElementCC3D("Steps", {}, gpd["MCS"])
        potts.ElementCC3D("Temperature", {}, gpd["MembraneFluctuations"])
        potts.ElementCC3D("NeighborOrder", {}, gpd["NeighborOrder"])

        if gpd["LatticeType"] != "Square":
            potts.ElementCC3D("LatticeType", {}, gpd["LatticeType"])

        for dim_name in ['x','y','z']:
            if gpd['BoundaryConditions'][dim_name] == 'Periodic':
                potts.ElementCC3D('Boundary_'+dim_name, {}, 'Periodic')

    def generateCellTypePlugin(self):
        """
        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Listing all cell types in the simulation")

        cell_type_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "CellType"})

        for type_id, typeTuple in self.idToTypeTupleDict.items():

            cell_type_dict = {}

            cell_type_dict["TypeName"] = typeTuple[0]
            cell_type_dict["TypeId"] = str(type_id)

            if typeTuple[1]:
                cell_type_dict["Freeze"] = ""
            cell_type_plugin_element.ElementCC3D("CellType", cell_type_dict)


    def generateContactPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies")

        contact_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Contact"})

        maxId = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, maxId + 1):
            for id2 in range(id1, maxId + 1):
                dict = {"Type1": self.idToTypeTupleDict[id1][0], "Type2": self.idToTypeTupleDict[id2][0]}
                contact_plugin_element.ElementCC3D("Energy", dict, 10)

        contact_plugin_element.ElementCC3D("NeighborOrder", {}, 4)



    def generateCompartmentPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies in the presence of compartmental cells")
        self.cc3d.addComment("Please note that this is obsolete way of handling compartmets")
        self.cc3d.addComment(
            "Please consider using ContactInternal contact to specify adhesions bewtween members of same cluster")

        compratment_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ContactCompartment"})

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            for id2 in range(id1, max_id + 1):

                cell_type_dict = {"Type1": self.idToTypeTupleDict[id1][0], "Type2": self.idToTypeTupleDict[id2][0]}
                compratment_plugin_element.ElementCC3D("Energy", cell_type_dict, 10)

        for id1 in range(0, max_id + 1):
            for id2 in range(id1, max_id + 1):

                cell_type_dict = {"Type1": self.idToTypeTupleDict[id1][0], "Type2": self.idToTypeTupleDict[id2][0]}
                compratment_plugin_element.ElementCC3D("InternalEnergy", cell_type_dict, 5)

        compratment_plugin_element.ElementCC3D("NeighborOrder", {}, 4)


    def generateContactInternalPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies in the presence of compartmental cells")
        self.cc3d.addComment(
            "Typically it is used in conjunction with \"reguar\" adhesion plugins e.g. Contact, AdhesionFlex etc...")

        contact_internal_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ContactInternal"})
        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(1, max_id + 1):
            for id2 in range(id1, max_id + 1):

                cell_type_dict = {"Type1": self.idToTypeTupleDict[id1][0], "Type2": self.idToTypeTupleDict[id2][0]}

                contact_internal_plugin_element.ElementCC3D("Energy", cell_type_dict, 5)

        contact_internal_plugin_element.ElementCC3D("NeighborOrder", {}, 4)

    def generateFocalPointPlasticityPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of focal point junctions")
        self.cc3d.addComment(
            "We separetely specify links between members of same cluster - "
            "InternalParameters and members of different clusters Parameters."
            " When not using compartmental  cells comment out InternalParameters specification")

        fppp_element = self.cc3d.ElementCC3D("Plugin", {"Name": "FocalPointPlasticity"})

        fppp_element.addComment("To modify FPP links individually for each cell pair uncomment line below")

        local_fppp_element = fppp_element.ElementCC3D("Local")
        local_fppp_element.commentOutElement()

        fppp_element.addComment(
            "Note that even hough you may manipulate lambdaDistance, targetDistance and maxDistance using Python "
            "you still need to set activation energy from XML level")

        fppp_element.addComment("See CC3D manual for details on FPP plugin ")

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(1, max_id + 1):
            for id2 in range(id1, max_id + 1):
                fppp_element.addComment("newline")
                cell_type_dict = {"Type1": self.idToTypeTupleDict[id1][0], "Type2": self.idToTypeTupleDict[id2][0]}

                param_element = fppp_element.ElementCC3D("Parameters", cell_type_dict)
                param_element.ElementCC3D("Lambda", {}, 10)
                param_element.ElementCC3D("ActivationEnergy", {}, -50)
                param_element.ElementCC3D("TargetDistance", {}, 7)
                param_element.ElementCC3D("MaxDistance", {}, 20)
                param_element.ElementCC3D("MaxNumberOfJunctions", {"NeighborOrder": 1}, 1)

        for id1 in range(1, max_id + 1):
            for id2 in range(id1, max_id + 1):
                fppp_element.addComment("newline")

                cell_type_dict = {"Type1": self.idToTypeTupleDict[id1][0], "Type2": self.idToTypeTupleDict[id2][0]}

                param_element = fppp_element.ElementCC3D("InternalParameters", cell_type_dict)
                param_element.ElementCC3D("Lambda", {}, 10)
                param_element.ElementCC3D("ActivationEnergy", {}, -50)
                param_element.ElementCC3D("TargetDistance", {}, 7)
                param_element.ElementCC3D("MaxDistance", {}, 20)
                param_element.ElementCC3D("MaxNumberOfJunctions", {"NeighborOrder": 1}, 1)

        fppp_element.addComment("newline")

        fppp_element.ElementCC3D("NeighborOrder", {}, 1)

    def generateElasticityPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Elastic constraints between Center of mass of cells. "
                             "Need to be accompanied by ElasticityTracker plugin to work. "
                             "Only cells in contact at MCS=0 will be affected by the constraint")

        self.cc3d.addComment( "ElasticityTracker keeps track of cell neighbors which are "
                              "participating in the elasticity constraint calculations")

        etp_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ElasticityTracker"})
        etp_element.addComment("Comment out cell types which should be unaffected by the constraint")

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            etp_element.ElementCC3D("IncludeType", {}, self.idToTypeTupleDict[id1][0])

        self.cc3d.addComment("newline")

        self.cc3d.addComment("This plugin calculates elastic constraints between cells Center of Mass")

        ep_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Elasticity"})

        ep_element.addComment( "To enable specification of elastic links individually for each "
                               "link uncomment line below ")

        local_elem = ep_element.ElementCC3D("Local")
        local_elem.commentOutElement()
        ep_element.addComment("See CC3D manual for details")

        ep_element.ElementCC3D("LambdaElasticity", {}, 200)
        ep_element.ElementCC3D("TargetLengthElasticity", {}, 6)

    def generateContactLocalProductPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies as a function of "
                             "cadherin concentration at cell membranes")
        self.cc3d.addComment("Adhesion energy is a function of two cells in ocntact. "
                             "the functional form is specified by the user")
        self.cc3d.addComment("Please consider using more flexible version of this plugin - AdhesionFlex")


        clp_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ContactLocalProduct"})

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            for id2 in range(id1, max_id + 1):
                cell_type_dict = {"Type1": self.idToTypeTupleDict[id1][0], "Type2": self.idToTypeTupleDict[id2][0]}

                clp_plugin_element.ElementCC3D("ContactSpecificity", cell_type_dict, -1)

        clp_plugin_element.ElementCC3D("ContactFunctionType", {}, "linear")
        clp_plugin_element.ElementCC3D("EnergyOffset", {}, 0.0)
        clp_plugin_element.ElementCC3D("NeighborOrder", {}, 4)

    def generateContactMultiCadPlugin(self):
        """

        :return:
        """

        cmc_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ContactMultiCad"})

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            for id2 in range(id1, max_id + 1):
                cell_type_dict = {"Type1": self.idToTypeTupleDict[id1][0], "Type2": self.idToTypeTupleDict[id2][0]}
                cmc_plugin_element.ElementCC3D("Energy", cell_type_dict, 10)

        specificity_cadherin_element = cmc_plugin_element.ElementCC3D("SpecificityCadherin")

        cad_max = len(self.cmcCadherins)

        for cad1 in range(cad_max):
            for cad2 in range(cad1, cad_max):

                cell_type_dict = {"Cadherin1": self.cmcCadherins[cad1], "Cadherin2": self.cmcCadherins[cad2]}
                specificity_cadherin_element.ElementCC3D("Specificity", cell_type_dict, -1)

        cmc_plugin_element.ElementCC3D("ContactFunctionType", {}, "linear")
        cmc_plugin_element.ElementCC3D("EnergyOffset", {}, 0.0)
        cmc_plugin_element.ElementCC3D("NeighborOrder", {}, 4)

    def generateAdhesionFlexPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment( "Specification of adhesion energies as a function of cadherin "
                              "concentration at cell membranes")

        self.cc3d.addComment("Adhesion energy is a function of two cells in ocntact. "
                             "the functional form is specified by the user")

        af_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "AdhesionFlex"})

        max_id = max(self.idToTypeTupleDict.keys())

        # listing adhesion molecules
        for molecule in self.afMolecules:
            af_plugin_element.ElementCC3D("AdhesionMolecule", {"Molecule": molecule})

        for id1 in range(0, max_id + 1):
            for molecule in self.afMolecules:
                af_plugin_element.ElementCC3D("AdhesionMoleculeDensity",
                                            {"CellType": self.idToTypeTupleDict[id1][0], "Molecule": molecule,
                                             "Density": 0.8})

        formula_element = af_plugin_element.ElementCC3D("BindingFormula", {"Name": "Binary"})

        formula_element.ElementCC3D("Formula", {}, self.afFormula)

        variables_element = formula_element.ElementCC3D("Variables")
        aim_element = variables_element.ElementCC3D("AdhesionInteractionMatrix")

        molecule_max = len(self.afMolecules)
        for molecule1 in range(molecule_max):
            for molecule2 in range(molecule1, molecule_max):

                cell_type_dict = {"Molecule1": self.afMolecules[molecule1], "Molecule2": self.afMolecules[molecule2]}
                aim_element.ElementCC3D("BindingParameter", cell_type_dict, -1.0)
        af_plugin_element.ElementCC3D("NeighborOrder", {}, 2)
    
    
    def generateImplicitMotilityPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Constraint on cell Motility. Each cell type has different constraint.")
        self.cc3d.addComment("For more flexible specification of the constraint (done in Python) "
                             "please use ImplicitMotilityLocal plugin")

        vf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ImplicitMotility"})

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            cell_type_dict = {"CellType": self.idToTypeTupleDict[id1][0],  "LambdaMotility": 15.0}

            vf_element.ElementCC3D("MotilityEnergyParameters", cell_type_dict)
    
    
    
    def generateVolumeFlexPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Constraint on cell volume. Each cell type has different constraint.")
        self.cc3d.addComment("For more flexible specification of the constraint (done in Python) "
                             "please use VolumeLocalFlex plugin")

        vf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Volume"})

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            cell_type_dict = {"CellType": self.idToTypeTupleDict[id1][0], "TargetVolume": 50, "LambdaVolume": 2.0}

            vf_element.ElementCC3D("VolumeEnergyParameters", cell_type_dict)

    def generateSurfaceFlexPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Constraint on cell surface. Each cell type has different constraint.")

        self.cc3d.addComment("For more flexible specification of the constraint "
                             "(done in Python) please use SurfaceLocalFlex plugin")

        sf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Surface"})

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            cell_type_dict = {"CellType": self.idToTypeTupleDict[id1][0], "TargetSurface": 30, "LambdaSurface": 0.5}

            sf_element.ElementCC3D("SurfaceEnergyParameters", cell_type_dict)

    def generateVolumeLocalFlexPlugin(self):
        """

        :return:
        """
        self.cc3d.addComment("newline")

        self.cc3d.addComment("Constraint on cell volume."
                             " Each cell has different constraint - constraints have "
                             "to be initialized and managed in Python")

        vf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Volume"})

    def generateSurfaceLocalFlexPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")

        self.cc3d.addComment("Constraint on cell surface. Each cell has different constraint - constraints "
                             "have to be initialized and managed in Python")

        sf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Surface"})

    def generateExternalPotentialPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("External force applied to cell. Each cell type has different force.")

        self.cc3d.addComment("For more flexible specification of the constraint (done in Python) please use "
                             "ExternalPotentialLocalFlex plugin")

        ep_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ExternalPotential"})
        ep_element.addComment("Algorithm options are: PixelBased, CenterOfMassBased")

        ep_element.ElementCC3D("Algorithm", {}, "PixelBased")

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            cell_type_dict = {"CellType": self.idToTypeTupleDict[id1][0], "x": -0.5, "y": 0.0, "z": 0.0}
            ep_element.ElementCC3D("ExternalPotentialParameters", cell_type_dict)

    def generateExternalPotentialLocalFlexPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("External force applied to cell. Each cell has different force and force "
                             "components have to be managed in Python.")

        self.cc3d.addComment("e.g. cell.lambdaVecX=0.5; cell.lambdaVecY=0.1 ; cell.lambdaVecZ=0.3;")

        ep_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ExternalPotential"})
        ep_element.addComment("Algorithm options are: PixelBased, CenterOfMassBased")
        ep_element.ElementCC3D("Algorithm", {}, "PixelBased")

    def generateConnectivityGlobalPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")

        self.cc3d.addComment( "Connectivity constraint applied to each cell. "
                              "Energy penalty specifies how strong the constraint is. "
                              "Penalty is specified for each type ")

        self.cc3d.addComment("This constraint works in 2D and 3D on all type of lattices. "
                             "It might be slowdown your simulation. For faster option - 2D and square lattice "
                             "you may use Connectivity or ConnectivityLocalFlex")

        self.cc3d.addComment("To speed up simulation comment out unnecessary constraints"
                             " for types which don't need the constraint")

        cp_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ConnectivityGlobal"})

        cp_element.addComment("By default we will always precheck connectivity "
                              "BUT in simulations in which there is no risk of having unfragmented cell "
                              "one can add this flag to speed up computations")

        cp_element.addComment("To turn off precheck uncomment line below")

        precheck_elem = cp_element.ElementCC3D("DoNotPrecheckConnectivity")
        precheck_elem.commentOutElement()

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            cell_type_dict = {"Type": self.idToTypeTupleDict[id1][0]}
            cp_element.ElementCC3D("Penalty", cell_type_dict, 1000000)

    def generateConnectivityGlobalByIdPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Connectivity constraint applied to each cell. "
                             "Energy penalty specifies how strong the constraint is. "
                             "Penalty is specified for each cell type individually ")

        self.cc3d.addComment("Use Python scripting to setup penalty (connectivity strength) for each cell")

        self.cc3d.addComment("e.g. self.connectivityGlobalPlugin.setConnectivityStrength(cell,20000000)"
                             " #cell, connectivity strength")

        self.cc3d.addComment("This constraint works in 2D and 3D on all type of lattices. "
                             "It might be slowdown your simulation. For faster option - 2D and "
                             "square lattice you may use Connectivity or ConnectivityLocalFlex")

        self.cc3d.addComment("To speed up simulation comment out unnecessary constraints "
                             "for types which don't need the constraint")

        cp_element = self.cc3d.ElementCC3D("Plugin", {"Name": "ConnectivityGlobal"})

    def generateConnectivityPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Connectivity constraint applied to each cell. "
                             "Energy penalty specifies how strong the constraint is. "
                             "Penalty is specified globally for each cell ")

        self.cc3d.addComment("This constraint works in 2D and on square lattice only!")

        self.cc3d.addComment("For more flexible version of this plugin use ConnectivityLocalFlex where "
                             "constraint penalty is specified for each cell individually using "
                             "Python scripting using the following syntax")

        self.cc3d.addComment("self.connectivityLocalFlexPlugin.setConnectivityStrength(cell,10000000)")

        cp_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Connectivity"})

        cp_element.ElementCC3D("Penalty", {}, 10000000)

    def generateLengthConstraintPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")

        self.cc3d.addComment("Applies elongation constraint to each cell. "
                             "Users specify target length of major axis -TargetLength (in 3D additionally, "
                             "target length of minor axis - MinorTargetLength) and a "
                             "strength of the constraint -LambdaLength. Parameters are specified for each cell type")

        self.cc3d.addComment("IMPORTANT: To prevent cell fragmentation for large elongations "
                             "you need to also use connectivity constraint")

        self.cc3d.addComment("LengthConstrainLocalFlex allows constrain specification for each cell "
                             "individually but currently works only in 2D")

        self.cc3d.addComment("Comment out the constrains for cell types which dont need them")

        lcp_element = self.cc3d.ElementCC3D("Plugin", {"Name": "LengthConstraint"})

        sim_3d_flag = self.checkIfSim3D()

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            cell_type_dict = {"CellType": self.idToTypeTupleDict[id1][0], "TargetLength": 20, "LambdaLength": 100}

            if sim_3d_flag:
                cell_type_dict["MinorTargetLength"] = 5

            lcp_element.ElementCC3D("LengthEnergyParameters", cell_type_dict)

    def generateLengthConstraintLocalFlexPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Applies elongation constraint to each cell. "
                             "Users specify the length major axis -TargetLength and a strength "
                             "of the constraint -LambdaLength. Parameters are specified for each cell individually")

        self.cc3d.addComment("IMPORTANT: To prevent cell fragmentation for large elongations "
                             "you need to also use connectivity constraint")

        self.cc3d.addComment("This plugin currently works only in 2D. "
                             "Use the following Python syntax to set/modify length constraint:")

        self.cc3d.addComment("self.lengthConstraintFlexPlugin.setLengthConstraintData(cell,20,30) "
                             " # cell , lambdaLength, targetLength  ")

        lcp_element = self.cc3d.ElementCC3D("Plugin", {"Name": "LengthConstraintLocalFlex"})

    def generateCenterOfMassPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module tracking center of mass of each cell")

        sf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "CenterOfMass"})

    def generateNeighborTrackerPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module tracking neighboring cells of each cell")
        sf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "NeighborTracker"})

    def generateMomentOfInertiaPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module tracking moment of inertia of each cell")
        sf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "MomentOfInertia"})

    def generatePixelTrackerPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module pixels of each cell")

        sf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "PixelTracker"})

    def generateBoundaryPixelTrackerPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module boundary pixels of each cell")

        sf_element = self.cc3d.ElementCC3D("Plugin", {"Name": "BoundaryPixelTracker"})
        sf_element.ElementCC3D("NeighborOrder", {}, 1)

    def generateSecretionPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of secretion properties of select cell types.")

        self.cc3d.addComment("Specification of secretion properties of individual cells can be done in Python")

        secretion_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Secretion"})

        for chem_field_name, secr_dict_list in self.secretionTable.items():

            secr_field_element = secretion_plugin_element.ElementCC3D("Field", {"Name": chem_field_name})

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
                    secr_field_element.ElementCC3D("SecretionOnContact", attribute_dict, rate)

    def generateChemotaxisPlugin(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of chemotaxis properties of select cell types.")

        # first translte chem field table into dictionary sorted by FieldName
        solver_dict = {}

        for field_tuple in self.chemFieldsTable:
            solver_dict[field_tuple[0]] = field_tuple[1]

        chemotaxis_plugin_element = self.cc3d.ElementCC3D("Plugin", {"Name": "Chemotaxis"})
        for chem_field_name, chem_dict_list in self.chemotaxisTable.items():

            chem_field_element = chemotaxis_plugin_element.ElementCC3D("ChemicalField",
                                                                   {"Name": chem_field_name})

            for chem_dict in chem_dict_list:

                lambda_ = chem_dict["Lambda"]
                chemotax_towards = chem_dict["ChemotaxTowards"]
                sat_coef = chem_dict["SatCoef"]
                chemotaxis_type = chem_dict["ChemotaxisType"]

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

    def generatePDESolverCaller(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module allowing multiple calls of the PDE solver. "
                             "By default number of extra calls is set to 0. ")

        self.cc3d.addComment("Change these settings to desired values after consulting CC3D manual on how to work with "
                             "large diffusion constants (>0.16 in 3D, >0.25 in 2D with DeltaX=1.0 and DeltaT=1.0)")

        # first translte chem field table into dictionary sorted by solver type
        solver_dict = {}

        for field_tuple in self.chemFieldsTable:
            try:
                solver_dict[field_tuple[1]].append(field_tuple[0])
            except LookupError:
                solver_dict[field_tuple[1]] = [field_tuple[0]]

        pc_elem = self.cc3d.ElementCC3D("Plugin", {"Name": "PDESolverCaller"})

        for solver in list(solver_dict.keys()):
            pc_elem.ElementCC3D("CallPDE", {"PDESolverName": solver, "ExtraTimesPerMC": 0})

    def generatePDESolvers(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of PDE solvers")

        # first translte chem field table into dictionary sorted by solver type
        solver_dict = {}

        for field_tuple in self.chemFieldsTable:
            try:
                solver_dict[field_tuple[1]].append(field_tuple[0])
            except LookupError:
                solver_dict[field_tuple[1]] = [field_tuple[0]]

        for solver, field_names in solver_dict.items():

            if solver == 'KernelDiffusionSolver':
                kdiff_solver_elem = self.cc3d.ElementCC3D("Steppable", {"Type": "KernelDiffusionSolver"})

                for field_name in field_names:

                    diff_field_elem = kdiff_solver_elem.ElementCC3D("DiffusionField", {"Name": field_name})
                    diff_field_elem.ElementCC3D("Kernel", {}, "4")
                    diff_data = diff_field_elem.ElementCC3D("DiffusionData")
                    diff_data.ElementCC3D("FieldName", {}, field_name)
                    diff_data.ElementCC3D("DiffusionConstant", {}, 0.1)
                    diff_data.ElementCC3D("DecayConstant", {}, 0.00001)

                    diff_data.addComment("Additional options are:")

                    do_not_diffuse_elem = diff_data.ElementCC3D("DoNotDiffuseTo", {}, "LIST YOUR CELL TYPES HERE")

                    do_not_diffuse_elem.commentOutElement()

                    do_not_decay_in_elem = diff_data.ElementCC3D("DoNotDecayIn", {}, "LIST YOUR CELL TYPES HERE")

                    do_not_decay_in_elem.commentOutElement()

                    conc_eqn_elem = diff_data.ElementCC3D("InitialConcentrationExpression", {}, "x*y")
                    conc_eqn_elem.commentOutElement()

                    conc_field_name_elem = diff_data.ElementCC3D("ConcentrationFileName", {},
                        "INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")

                    conc_field_name_elem.commentOutElement()

                    diff_data.ElementCC3D("DeltaX", {}, 1.0)
                    diff_data.ElementCC3D("DeltaT", {}, 1.0)

                    # add commented out concentration field specification file

            elif solver in ('FlexibleDiffusionSolverFE', 'FastDiffusionSolver2DFE'):
                diff_solver_elem = self.cc3d.ElementCC3D("Steppable", {"Type": solver})

                for field_name in field_names:
                    diff_field_elem = diff_solver_elem.ElementCC3D("DiffusionField", {"Name": field_name})
                    diff_data = diff_field_elem.ElementCC3D("DiffusionData")
                    diff_data.ElementCC3D("FieldName", {}, field_name)
                    diff_data.ElementCC3D("DiffusionConstant", {}, 0.1)
                    diff_data.ElementCC3D("DecayConstant", {}, 0.00001)

                    diff_data.addComment("Additional options are:")

                    do_not_diffuse_elem = diff_data.ElementCC3D("DoNotDiffuseTo", {}, "LIST YOUR CELL TYPES HERE")

                    do_not_diffuse_elem.commentOutElement()

                    do_not_decay_in_elem = diff_data.ElementCC3D("DoNotDecayIn", {}, "LIST YOUR CELL TYPES HERE")

                    do_not_decay_in_elem.commentOutElement()

                    conc_eqn_elem = diff_data.ElementCC3D("InitialConcentrationExpression", {}, "x*y")

                    conc_eqn_elem.commentOutElement()

                    conc_field_name_elem = diff_data.ElementCC3D("ConcentrationFileName", {},
                             "INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")

                    conc_field_name_elem.commentOutElement()

                    diff_data.addComment("To run solver for large diffusion constants you typically "
                                         "call solver multiple times - ExtraTimesPerMCS to specify additional calls "
                                         "to the solver in each MCS ")

                    diff_data.addComment("IMPORTANT: make sure not to mix this setting with the "
                                         "PDESolverCaller module! See manual for more information")

                    extra_times_per_mcs_elem = diff_data.ElementCC3D("ExtraTimesPerMCS", {}, 0)
                    extra_times_per_mcs_elem.commentOutElement()

                    diff_data.ElementCC3D("DeltaX", {}, 1.0)
                    diff_data.ElementCC3D("DeltaT", {}, 1.0)

            elif solver in ('SteadyStateDiffusionSolver'):
                solver_name = 'SteadyStateDiffusionSolver2D'

                sim_3d_flag = self.checkIfSim3D()

                if sim_3d_flag:
                    solver_name = 'SteadyStateDiffusionSolver'

                diff_solver_elem = self.cc3d.ElementCC3D("Steppable", {"Type": solver_name})

                for field_name in field_names:

                    diff_field_elem = diff_solver_elem.ElementCC3D("DiffusionField", {"Name": field_name})

                    diff_data = diff_field_elem.ElementCC3D("DiffusionData")
                    diff_data.ElementCC3D("FieldName", {}, field_name)
                    diff_data.ElementCC3D("DiffusionConstant", {}, 1.0)
                    diff_data.ElementCC3D("DecayConstant", {}, 0.00001)
                    conc_eqn_elem = diff_data.ElementCC3D("InitialConcentrationExpression", {}, "x*y")
                    conc_eqn_elem.commentOutElement()
                    conc_field_name_elem = diff_data.ElementCC3D("ConcentrationFileName", {},
                         "INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")

                    conc_field_name_elem.commentOutElement()

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

            else:
                return


    def generateBoxWatcherSteppable(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment( "Module tracing boundaries of the minimal box enclosing all the cells. "
                              "May speed up calculations. May have no effect for parallel version")

        bw_elem = self.cc3d.ElementCC3D("Steppable", {"Type": "BoxWatcher"})
        bw_elem.ElementCC3D("XMargin", {}, 7)
        bw_elem.ElementCC3D("YMargin", {}, 7)
        bw_elem.ElementCC3D("ZMargin", {}, 7)

    def generateUniformInitializerSteppable(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Initial layout of cells in the form of rectangular slab")

        ui_element = self.cc3d.ElementCC3D("Steppable", {"Type": "UniformInitializer"})

        region = ui_element.ElementCC3D("Region")

        gpd = self.generalPropertiesDict

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

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            types_string += self.idToTypeTupleDict[id1][0]

            if id1 < max_id:
                types_string += ","

        region.ElementCC3D("Types", {}, types_string)


    def generateBlobInitializerSteppable(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Initial layout of cells in the form of spherical (circular in 2D) blob")

        ui_element = self.cc3d.ElementCC3D("Steppable", {"Type": "BlobInitializer"})
        region = ui_element.ElementCC3D("Region")

        gpd = self.generalPropertiesDict

        x_center = int(gpd["Dim"][0] / 2)
        y_center = int(gpd["Dim"][1] / 2)
        z_center = int(gpd["Dim"][2] / 2)

        max_dim = max([x_center, y_center, z_center])

        region.ElementCC3D("Center", {"x": x_center, "y": y_center, "z": z_center})
        region.ElementCC3D("Radius", {}, int(max_dim / 2.5))
        region.ElementCC3D("Gap", {}, 0)
        region.ElementCC3D("Width", {}, 7)

        types_string = ""

        max_id = max(self.idToTypeTupleDict.keys())

        for id1 in range(0, max_id + 1):
            if self.idToTypeTupleDict[id1][0] == "Medium":
                continue

            types_string += self.idToTypeTupleDict[id1][0]

            if id1 < max_id:
                types_string += ","

        region.ElementCC3D("Types", {}, types_string)

    def generatePIFFInitializerSteppable(self):
        """

        :return:
        """

        self.cc3d.addComment("newline")
        self.cc3d.addComment("Initial layout of cells using PIFF file. Piff files can be generated using PIFGEnerator")

        ui_element = self.cc3d.ElementCC3D("Steppable", {"Type": "PIFInitializer"})

        gpd = self.generalPropertiesDict
        if gpd["Initializer"][0] == "piff":
            ui_element.ElementCC3D("PIFName", {}, gpd["Initializer"][1])
        else:
            ui_element.ElementCC3D("PIFName", {}, "PLEASE_PUT_PROPER_FILE_NAME_HERE")

    def generatePIFFDumperSteppable(self):
        """

        :return:
        """
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Periodically stores cell layout configuration in a piff format")

        pifd_element = self.cc3d.ElementCC3D("Steppable", {"Type": "PIFDumper", "Frequency": 100})

        pifd_element.ElementCC3D("PIFName", {}, self.simulationName)

        pifd_element.ElementCC3D("PIFFileExtension", {}, "piff")


    def saveCC3DXML(self):
        """

        :return:
        """

        self.cc3d.CC3DXMLElement.saveXML(str(self.fileName))
        # self.cc3d.CC3DXMLElement.saveXMLInPython(str(self.fileName+".py"))
        print("SAVING XML = ", self.fileName)
        # print "SAVING XML in Python= ",self.fileName+".py"

