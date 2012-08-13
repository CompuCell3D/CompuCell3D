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
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR1 PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#ifndef CONTACTLOCALFLEXDATA_H
#define CONTACTLOCALFLEXDATA_H

#include "ContactLocalFlexDLLSpecifier.h"
#include <set>
#include <vector>


namespace CompuCell3D {

class CellG;


class CONTACTLOCALFLEX_EXPORT ContactLocalFlexData{

   public:
      ContactLocalFlexData():neighborAddress(0),J(0){}
      ///have to define < operator if using a class in the set and no < operator is defined for this class
      bool operator<(const ContactLocalFlexData & _rhs) const{
         return neighborAddress < _rhs.neighborAddress;
      }
	  
      CellG * neighborAddress;
      double J;	  

};


   class ContactLocalFlexDataContainer{
      public:
         ContactLocalFlexDataContainer(){};
         ~ContactLocalFlexDataContainer(){};
         std::set<ContactLocalFlexData> contactDataContainer; //stores contact energies for cell neighbors
         //add local default values
         std::vector< std::vector<double> > localDefaultContactEnergies;
         
   };







};
#endif
