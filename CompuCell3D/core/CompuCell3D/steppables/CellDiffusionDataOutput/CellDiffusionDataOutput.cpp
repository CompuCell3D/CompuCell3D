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

#include "CellDiffusionDataOutput.h"

#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Automaton/Automaton.h>

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
using namespace CompuCell3D;

#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>


#include <string>

#include <plugins/CenterOfMass/CenterOfMassPlugin.h>
#include <fstream>
#include <sstream>

using namespace std;


CellDiffusionDataOutput::CellDiffusionDataOutput() :
  potts(0)
{

    cellIDFlag=false;
    deltaPositionFlag=false;

}

CellDiffusionDataOutput::~CellDiffusionDataOutput(){

      for(int i = 0 ; i < filePtrVec.size() ; ++i){
         if(filePtrVec[i]){

            filePtrVec[i]->close();
            delete filePtrVec[i];
            filePtrVec[i]=0;

         }
      }

}

void CellDiffusionDataOutput::init(Simulator *simulator) {
  potts = simulator->getPotts();
  cellInventoryPtr=& potts->getCellInventory();
}

void CellDiffusionDataOutput::extraInit(Simulator *simulator) {

   


}


//this will work properly when initialization is called before this steppable
void CellDiffusionDataOutput::start() {

   //initialize streams

      CenterOfMassPlugin * centerOfMassPluginPtr=(CenterOfMassPlugin*)(Simulator::pluginManager.get("CenterOfMass"));
      ASSERT_OR_THROW("CenterOfMass plugin not initialized!", centerOfMassPluginPtr);
      //now will assign cellPtrs based on cell ids - this will make cell search in inventory much faster
      CellInventory::cellInventoryIterator cInvItr;
      CellG * cell;
      
      for (int i = 0 ; i < cellIds.size() ; ++i ){
         cInvItr=cellInventoryPtr->find(cellIds[i]);
         if(cInvItr != cellInventoryPtr->cellInventoryEnd()){
            cell=*cInvItr;
            cellIdsPtrs.push_back(cell);
         }else{
            cerr<<"Could not find in the inventory cell with id="<<cellIds[i]<<" . Ignoring request"<<endl;
         }
      }

      //allocate memory for "previous CM" vector
      cellPositions.assign(cellIdsPtrs.size(),Coordinates3D<float>());
      //allocate memory for out streams
      filePtrVec.assign(cellIdsPtrs.size(),0);
      for(int i = 0 ; i < filePtrVec.size() ; ++i){
         ostringstream str;
         str<<fileName<<"_"<<cellIdsPtrs[i]->id<<".txt";
         filePtrVec[i]=new ofstream(str.str().c_str());
         cerr<<"Creating file name="<<str.str()<<endl;
      }

   //initializing initial cell positions

   float xCM,yCM,zCM;
//   CellG * cell;

   if(deltaPositionFlag){
      for(int i = 0 ; i < cellPositions.size() ; ++i ){

         if( cellInventoryPtr->find(cellIdsPtrs[i]) == cellInventoryPtr->cellInventoryEnd() ){
            continue;//this means that this cell is no longer in the inventory
         }

         cell= cellIdsPtrs[i];
         xCM=cell->xCM/(float)cell->volume;
         yCM=cell->yCM/(float)cell->volume;
         zCM=cell->zCM/(float)cell->volume;
         cellPositions[i]=Coordinates3D<float>(xCM,yCM,zCM);

      }
   }

}



void CellDiffusionDataOutput::step(const unsigned int currentStep) {

   float xCM,yCM,zCM;
   CellG * cell;
   float deltaPos;
   Point3D prevCM;

   for(int i = 0 ; i < cellIdsPtrs.size() ; ++i){
      ofstream &out=(*filePtrVec[i]);
      if( cellInventoryPtr->find(cellIdsPtrs[i]) == cellInventoryPtr->cellInventoryEnd() ){
         filePtrVec[i]->close();
         continue;//this means that this cell is no longer in the inventory
      }
      cell= cellIdsPtrs[i];

      xCM=cell->xCM/(float)cell->volume;
      yCM=cell->yCM/(float)cell->volume;
      zCM=cell->zCM/(float)cell->volume;

      out<<cell->id<<"\t";
      out<<currentStep<<"\t";
      out<<xCM<<"\t"<<yCM<<"\t"<<zCM<<"\t";

      if(deltaPositionFlag){

         prevCM.x=cellPositions[i].X();
         prevCM.y=cellPositions[i].Y();
         prevCM.z=cellPositions[i].Z();
         deltaPos=sqrt((xCM-prevCM.x)*(xCM-prevCM.x)+(yCM-prevCM.y)*(yCM-prevCM.y)+(zCM-prevCM.z)*(zCM-prevCM.z));
         //cellPositions[i]=Coordinates3D<float>(xCM,yCM,zCM);
         out<<deltaPos<<"\t";
      }

      out<<endl;
   }







}

void CellDiffusionDataOutput::readXML(XMLPullParser &in) {

   in.skip(TEXT);

  string type;
  
  while (in.check(START_ELEMENT)) {

    if(in.getName() == "OutputFormat"){

      if(in.findAttribute("DeltaPosition")>=0){

            deltaPositionFlag=true;
      }

      if(in.findAttribute("FileName")>=0){
         fileName=in.getAttribute("FileName").value;;
      }else{
         fileName="Output";
      }

      in.matchSimple();
   }

    else if (in.getName() == "OutputID") {
      

      if(in.findAttribute("CellID")>=0){
         unsigned int id=BasicString::parseUInteger(in.getAttribute("CellID").value);
         cellIds.push_back(id);
      }

      

      in.matchSimple();

    }else {
      throw BasicException(string("Unexpected element '") + in.getName() + 
			   "'!", in.getLocation());
    }

    in.skip(TEXT);
  }
}

void CellDiffusionDataOutput::writeXML(XMLSerializer &out) {
}







