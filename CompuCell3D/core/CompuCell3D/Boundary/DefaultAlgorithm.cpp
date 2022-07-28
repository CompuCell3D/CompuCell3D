#include "Algorithm.h"
#include "DefaultAlgorithm.h"

using namespace CompuCell3D;

/*
 * Read the input file and populate
 * our 3D vector.
 * @ return void.
 */
void DefaultAlgorithm::readFile(const int index, const int size, string
inputfile) {}


/*
 * Apply default algorithm.
 * Return 'true' if the passed point is in the grid.
 *
 */
bool DefaultAlgorithm::inGrid(const Point3D &pt) {
    return (0 <= pt.x && pt.x < dim.x &&
            0 <= pt.y && pt.y < dim.y &&
            0 <= pt.z && pt.z < dim.z);
}


/*
 * Get Number of Cells 
 * 
 * @param x  int
 * @param y  int
 * @param z  int
 *
 * @return int
 *
 */
int DefaultAlgorithm::getNumPixels(int x, int y, int z) {

    return x * y * z;

}
