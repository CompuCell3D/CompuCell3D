#include "Boundary.h"
#include "PeriodicBoundary.h"
#include <cmath>
#include <iostream>

using namespace std;

using namespace CompuCell3D;


/*
 * Apply PeriodicBoundary to the given coordinate. 
 * If the coordinate lies outside the max value take a mod and return it/
 * If the coordinate is negative, take a mod and subtract that value
 * from the max value and return it 
 *
 * @param coordinate  int
 * @param max_value int
 * 
 * @return bool. If the condition was applied successfully.
 */
bool PeriodicBoundary::applyCondition(int &coordinate, const int &max_value) {

    short val;

    if (coordinate < 0) {


        val = abs((float) (coordinate % max_value));
        coordinate = max_value - val;
        return true;


    } else if (coordinate >= max_value) {


        coordinate = coordinate % max_value;
        return true;

    }

    return false;


}

