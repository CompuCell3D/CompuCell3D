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

#ifndef SIMOBJECT_H
#define SIMOBJECT_H


#include <string>
#include <CompuCell3D/SteerableObject.h>

class CC3DXMLElement;

namespace CompuCell3D {
  class Simulator;
  class ParseData;
  class CC3DEvent;

  class SimObject : public virtual SteerableObject {
  protected:
    Simulator * simulator;
    ParseData * pd;
  public:
    SimObject() : simulator(0),pd(0) {}
    virtual ~SimObject() {}

    
    virtual void init(Simulator *simulator,  CC3DXMLElement * _xmlData=0) {this->simulator = simulator;}
    virtual void extraInit(Simulator *simulator){this->simulator = simulator;}
    virtual std::string toString(){return "SimObject";}
    virtual ParseData * getParseData(){return pd;}
    virtual void handleEvent(CC3DEvent & _event){}

  };
}
#endif
