#include <CompuCell3D/CC3D.h>

#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>

using namespace CompuCell3D;
using namespace std;


#include "ChemotaxisSimpleEnergy.h"

float ChemotaxisSimpleEnergy::simpleChemotaxisFormula(float _flipNeighborConc, float _conc, double _lambda) {
    return (_flipNeighborConc - _conc) * _lambda;
}




