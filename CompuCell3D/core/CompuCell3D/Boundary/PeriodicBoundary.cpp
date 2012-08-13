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
  bool PeriodicBoundary::applyCondition(int& coordinate, const int& max_value){
  
      short val;
      
      if(coordinate < 0 ) {


          val = abs((float)(coordinate % max_value));
          coordinate = max_value - val;
          return true;


     } else if (coordinate >= max_value) {

          
          coordinate = coordinate % max_value;
          return true;

     }
      
      return false;


  }

