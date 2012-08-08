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

#ifndef CHENGBANGALGORITHM_H
#define CHENGBANGALGORITHM_H

#include "Algorithm.h"

#include <vector>
using std::vector;

namespace CompuCell3D {

    /*
     * Chengbang's Algorithm. 
     */
   class ChengbangAlgorithm : public Algorithm {


        public:       
          ChengbangAlgorithm() {evolution=-1;}
          void readFile(const int index, const int size, string inputfile);
          bool inGrid(const Point3D& pt);
          int getNumPixels(int x, int y, int z); 
          int i;
          int s;
          string filetoread;
	  int evolution;
        private:
         vector<vector<vector<float> > > dataStructure;
          void readFile(const char* inputFile);
   };
    
};




#endif
