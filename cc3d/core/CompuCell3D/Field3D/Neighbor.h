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

#ifndef NEIGHBOR_H
#define NEIGHBOR_H

#include "../Field3D/Point3D.h"
#include <Utils/Coordinates3D.h>
#include <iostream>

namespace CompuCell3D {

  /** 
   * Used by NeighborFinder to hold the offset to a neighbor Point3D and
   * it's distance.
   */
  class Neighbor {
  public:
    Point3D pt;
    double distance;
    Coordinates3D<double> ptTrans;

    Neighbor() : distance(0) {}
    Neighbor(const Point3D pt, const double distance,const Coordinates3D<double> _ptTrans=Coordinates3D<double>(.0,.0,.0)) :
      pt(pt), distance(distance),ptTrans(_ptTrans) {}
  };
  

   inline std::ostream & operator<<(std::ostream & _scr,const Neighbor & _n){
      using namespace std;
      _scr<<"pt="<<_n.pt<<" ptTrans="<<_n.ptTrans<<" distance="<<_n.distance;
      return _scr;

   }
};
#endif
