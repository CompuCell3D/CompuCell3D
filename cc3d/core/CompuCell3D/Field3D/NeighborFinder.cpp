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

#include "NeighborFinder.h"
using namespace CompuCell3D;

#include <math.h>

NeighborFinder *NeighborFinder::singleton = 0;

void NeighborFinder::getMore() {
  int nums[3];
  int sqrs[3];
  int sumSqrs;
  depth++;
  
  for (nums[0] = 0; true; nums[0]++) {
    sqrs[0] = nums[0] * nums[0];
    if (sqrs[0] > depth) break;

    for (nums[1] = 0; nums[1] <= nums[0]; nums[1]++) {
      sqrs[1] = nums[1] * nums[1];
      if (sqrs[1] > depth - sqrs[0]) break;
      
      for (nums[2] = 0; nums[2] <= nums[1]; nums[2]++) {
	sqrs[2] = nums[2] * nums[2];
	if (sqrs[2] > depth - sqrs[0] - sqrs[1]) break;

	sumSqrs = sqrs[0] + sqrs[1] + sqrs[2];
	if (sumSqrs == depth) addNeighbors(nums, sqrt((double)sumSqrs));
      }
    }
  }
}

void NeighborFinder::addNeighbors(int nums[3], const double distance) {

  
  Point3D pt;
  bool firstX;
  bool firstY;
  bool firstZ;

  firstX = true;
  for (int x = 0; x < 3; x++) {
    if (!firstX && nums[x] == pt.x) continue;
    pt.x = nums[x];
    firstX = false;

    firstY = true;
    for (int y = 0; y < 3; y++) {
      if (!firstY && nums[y] == pt.y) continue;
      if (y == x) continue;
      pt.y = nums[y];
      firstY = false;
      
      firstZ = true;
      for (int z = 0; z < 3; z++) {
	if (!firstZ && nums[z] == pt.z) continue;
	if (z == x || z == y) continue;
	pt.z = nums[z];
	firstZ = false;

	while (true) {
	  while (true) {
	    while (true) {
	      neighbors.put(Neighbor(pt, distance));

	      pt.z = -pt.z;
	      if (pt.z >= 0) break;
	    }
	    pt.y = -pt.y;
	    if (pt.y >= 0) break;
	  }
	  pt.x = -pt.x;
	  if (pt.x >= 0) break;
	}
      }
    }
  }
}

NeighborFinder::~NeighborFinder()
{
	singleton = 0;
}

void NeighborFinder::destroy()
{
	if(singleton)
		delete singleton;
}

