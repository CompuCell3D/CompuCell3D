

 #include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/ClassRegistry.h>
// // // #include <CompuCell3D/Simulator.h>
// // // //#include <CompuCell3D/Diffusable.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/Field3DIO.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // //#include <CompuCell3D/DiffusionSolverBiofilmFE.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>
using namespace CompuCell3D;


// // // #include <PublicUtilities/StringUtils.h>


// // // #include <fstream>
// // // #include <string>
using namespace std;


#include "ChemotaxisSimpleEnergy.h"

float ChemotaxisSimpleEnergy::simpleChemotaxisFormula(float _flipNeighborConc,float _conc,double _lambda){
   return (_flipNeighborConc-_conc)*_lambda;
}




