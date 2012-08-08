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



#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
using namespace CompuCell3D;


#include <CompuCell3D/Potts3D/CellInventory.h>

#include <iostream>
using namespace std;


#include "VolumeMean.h"

VolumeMean::VolumeMean() :  cellFieldG(0),sim(0),potts(0),exponent(1.0) {}

VolumeMean::~VolumeMean() {
}


void VolumeMean::init(Simulator *simulator, CC3DXMLElement *_xmlData) {


  potts = simulator->getPotts();
  sim=simulator;
  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();


  simulator->registerSteerableObject(this);
  update(_xmlData);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeMean::extraInit(Simulator *simulator){

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void VolumeMean::start(){

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeMean::step(const unsigned int currentStep){

   CellInventory *cellInventoryPtr=& potts->getCellInventory();
   CellInventory::cellInventoryIterator cInvItr;
   CellG *cell;
   double mean=0.0;
   unsigned int cellCounter=0;
   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
      cell=cellInventoryPtr->getCell(cInvItr);
      // cell=*cInvItr;
      mean+=pow((double)cell->volume,(double)exponent);
      ++cellCounter;


   }
   if(cellCounter)
      mean/=cellCounter;
   else
      mean=0.0;

   cerr<<"The mean cell volume for exponent of "<<exponent<<" is "<<mean<<endl;

}






//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeMean::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

     if(_xmlData->findElement("Exponent"))
        exponent=_xmlData->getFirstElement("Exponent")->getDouble();

}

std::string VolumeMean::toString(){
   return "VolumeMean";
}


std::string VolumeMean::steerableName(){
   return toString();
}


