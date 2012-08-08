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

#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <string>

using namespace std;

#include <CompuCell3D/Field3D/Dim3D.h>

namespace CompuCell3D {
   class Point3D;
   /*
     * Interface for Algorithm
     */
   class Algorithm {


       public:
           virtual ~Algorithm() {}
           virtual void readFile(const int index, const int size, string inputfile)=0;
           virtual bool inGrid(const Point3D& pt)=0;
           void setDim(Dim3D theDim) {dim = theDim;}
           void setCurrentStep(int theCurrentStep) {currentStep = theCurrentStep;}
           virtual int getNumPixels(int x, int y, int z)=0 ;
       protected:
           Dim3D dim;
           int currentStep;
   }; 
    
};

#endif
