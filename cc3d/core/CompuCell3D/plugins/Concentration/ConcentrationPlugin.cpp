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

#include "ConcentrationPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
using namespace CompuCell3D;

#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include <iostream>
using namespace std;

ConcentrationPlugin::ConcentrationPlugin() : potts(0) {}

ConcentrationPlugin::~ConcentrationPlugin() {}

void ConcentrationPlugin::init(Simulator *simulator) {
   potts = simulator->getPotts();
   
   potts->getCellFactoryGroupPtr()->registerClass(&concentrationAccessor);
  
   potts->registerCellGChangeWatcher(this);

   potts->registerStepper(this);
  
  
}

void ConcentrationPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {
   
   ///although currently main updating of cumulative concentration and decision whether to kill cell is made in target volume steppable
   ///we still may need at some poit to start making the decision in this plugin  so I leave code that updates cumulative concentration here
   ///It will be just comented out
//    if(oldCell){
//       concentrationAccessor.get(oldCell->extraAttribPtr)->concentration -= concentrationFieldPtr->get(pt);
// 
// 
//    }
// 
//    if(newCell){
// 
//       concentrationAccessor.get(newCell->extraAttribPtr)->concentration += concentrationFieldPtr->get(pt);
//    }


}


void ConcentrationPlugin::step() {
}

void ConcentrationPlugin::readXML(XMLPullParser &in) {

}

void ConcentrationPlugin::writeXML(XMLSerializer &out) {

}
