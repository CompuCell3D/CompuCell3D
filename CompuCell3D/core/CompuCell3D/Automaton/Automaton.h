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

#ifndef AUTOMATON_H
#define AUTOMATON_H

//#include <CompuCell3D/Potts3D/CellChangeWatcher.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include "CellType.h"

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h>
#include <CompuCell3D/plugins/CellType/CellTypeG.h>
#include <iostream>

#include <string>
//using namespace std;

namespace CompuCell3D {
  class Potts3D;
  class Cell;
  
  class Automaton : public CellGChangeWatcher {
      
  protected:
    Potts3D* potts;
    
    CellType* classType;

  public:
    Automaton() {}
    
    virtual ~Automaton() {if (classType) delete classType;};

    
    virtual void creation(CellG* newCell) {}
    virtual void updateVariables(CellG* newCell) {}

    virtual void field3DChange(const Point3D &pt, CellG *newCell,
			       CellG *oldCell);



    virtual unsigned char getCellType(const CellG*) const =0;
    virtual std::string getTypeName(const char type) const =0;
    virtual unsigned char getTypeId(const std::string typeName) const =0;
	virtual unsigned char getMaxTypeId() const =0;
  };
};
#endif
