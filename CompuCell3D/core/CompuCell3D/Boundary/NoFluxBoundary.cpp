#include "Boundary.h"
#include "NoFluxBoundary.h"

using namespace CompuCell3D;

/*
 * Apply NoFluxBoundary to the given coordinate. Since NoFlux boundary condition
 * essentially is no condition. It returns false.
 *
 * @param coordinate  int
 * @param max_value int
 * 
 * @return bool. If the condition was applied successfully.
 */
bool NoFluxBoundary::applyCondition(int &coordinate, const int &max_value) {

    return false;
}
 

