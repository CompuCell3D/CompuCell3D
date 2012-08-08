// ************************************************************
// Module Includes
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.


%{


// CompuCell3D Include Files
#include <CompuCell3D/ClassRegistry.h>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>
#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
//NeighborFinderParams
#include <NeighborFinderParams.h>
#include <Python.h>
#include <Utils/Coordinates3D.h>


// Third Party Libraries

// System Libraries
#include <iostream>
#include <stdlib.h>




//PyObjectAdapetrs

//EnergyFcns
//#include <CompuCell3D/Potts3D/EnergyFunction.h>

#include <PyCompuCellObjAdapter.h>
#include <EnergyFunctionPyWrapper.h>
#include <ChangeWatcherPyWrapper.h>
#include <TypeChangeWatcherPyWrapper.h>
#include <StepperPyWrapper.h>
#include <PyAttributeAdder.h>


#include <CompuCell3D/ParseData.h>
#include <CompuCell3D/ParserStorage.h>

//Plugins

#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation

#include <CompuCell3D/Automaton/Automaton.h>

#include <CompuCell3D/plugins/ConnectivityLocalFlex/ConnectivityLocalFlexData.h>

#include <CompuCell3D/plugins/ConnectivityLocalFlex/ConnectivityLocalFlexPlugin.h>

#include <CompuCell3D/plugins/LengthConstraintLocalFlex/LengthConstraintLocalFlexData.h>

#include <CompuCell3D/plugins/LengthConstraintLocalFlex/LengthConstraintLocalFlexPlugin.h>

#include <CompuCell3D/plugins/ChemotaxisSimple/ChemotaxisSimpleEnergy.h>

#include <CompuCell3D/plugins/Chemotaxis/ChemotaxisData.h>
#include <CompuCell3D/plugins/Chemotaxis/ChemotaxisPlugin.h>


// //plugins
#include <CompuCell3D/plugins/Mitosis/MitosisPlugin.h>

#include <CompuCell3D/plugins/Mitosis/MitosisSimplePlugin.h>

#include <CompuCell3D/plugins/NeighborTracker/NeighborTracker.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>

#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>

#include <CompuCell3D/plugins/PixelTracker/BoundaryPixelTracker.h>
#include <CompuCell3D/plugins/PixelTracker/BoundaryPixelTrackerPlugin.h>

#include <CompuCell3D/plugins/ContactLocalFlex/ContactLocalFlexData.h>
#include <CompuCell3D/plugins/ContactLocalFlex/ContactLocalFlexPlugin.h>

#include <CompuCell3D/plugins/ContactLocalProduct/ContactLocalProductData.h>
#include <CompuCell3D/plugins/ContactLocalProduct/ContactLocalProductPlugin.h>

#include <CompuCell3D/plugins/ContactMultiCad/ContactMultiCadData.h>
#include <CompuCell3D/plugins/ContactMultiCad/ContactMultiCadPlugin.h>

#include <CompuCell3D/plugins/AdhesionFlex/AdhesionFlexData.h>
#include <CompuCell3D/plugins/AdhesionFlex/AdhesionFlexPlugin.h>


// #include <CompuCell3D/plugins/MolecularContact/MolecularContactPlugin.h>

#include <CompuCell3D/plugins/CellOrientation/CellOrientationVector.h>
#include <CompuCell3D/plugins/CellOrientation/CellOrientationPlugin.h>

#include <CompuCell3D/plugins/PolarizationVector/PolarizationVector.h>
#include <CompuCell3D/plugins/PolarizationVector/PolarizationVectorPlugin.h>

#include <CompuCell3D/plugins/Elasticity/ElasticityTracker.h>
#include <CompuCell3D/plugins/Elasticity/ElasticityTrackerPlugin.h>
      
#include <CompuCell3D/plugins/Plasticity/PlasticityTracker.h>
#include <CompuCell3D/plugins/Plasticity/PlasticityTrackerPlugin.h>

#include <CompuCell3D/plugins/FocalPointPlasticity/FocalPointPlasticityTracker.h>
#include <CompuCell3D/plugins/FocalPointPlasticity/FocalPointPlasticityPlugin.h>

#include <CompuCell3D/plugins/MomentOfInertia/MomentOfInertiaPlugin.h>

#include <CompuCell3D/plugins/Secretion/FieldSecretor.h>
#include <CompuCell3D/plugins/Secretion/SecretionPlugin.h>

//Steppables
#include <CompuCell3D/steppables/Mitosis/MitosisSteppable.h>

// Namespaces
using namespace std;
using namespace CompuCell3D;

#define SWIG_EXPORT_ITERATOR_METHODS

%}



