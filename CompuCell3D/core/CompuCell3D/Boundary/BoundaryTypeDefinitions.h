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

#ifndef BOUNDARYTYPEDEFINITIONS_H
#define BOUNDARYTYPEDEFINITIONS_H

#include <string>

//using namespace std;

namespace CompuCell3D {

   enum LatticeType {SQUARE_LATTICE=1,HEXAGONAL_LATTICE=2};    

   class LatticeMultiplicativeFactors{
      public:
         LatticeMultiplicativeFactors(double _volumeMF=1.0, double _surfaceMF=1.0, double _lengthMF=1.0):
         volumeMF(_volumeMF),
         surfaceMF(_surfaceMF),
         lengthMF(_lengthMF)
         {}
         double volumeMF;
         double surfaceMF;
         double lengthMF;
   };

};

#endif
