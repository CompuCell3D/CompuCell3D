/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/


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
bool DefaultAlgorithm::inGrid(const Point3D& pt) {
         return (0 <= pt.x && pt.x < dim.x &&
                   0 <= pt.y && pt.y < dim.y &&
                   0 <= pt.z && pt.z < dim.z );
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

    return x*y*z; 

}
