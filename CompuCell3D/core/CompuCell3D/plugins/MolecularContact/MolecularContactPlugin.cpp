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

#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <PublicUtilities/NumericalUtils.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <ctime>

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
using namespace CompuCell3D;


#include <iostream>
using namespace std;

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <algorithm>


//#define EXP_STL
#include "MolecularContactPlugin.h"

#include <Python.h>
#include<core/CompuCell3D/CC3DLogger.h>


MolecularContactPlugin::MolecularContactPlugin() :
cellFieldG(0),
targetLengthMolecularContact(0.0),
lambdaMolecularContact(0.0),
maxLengthMolecularContact(100000000000.0),
boundaryStrategy(0),initialized(false)
{}

MolecularContactPlugin::~MolecularContactPlugin() {

}

void MolecularContactPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

   potts = simulator->getPotts();
   cellFieldG = potts->getCellFieldG();
   sim = simulator;

   ///getting cell inventory
   cellInventoryPtr=& potts->getCellInventory();

   mcsState = -1;
   potts->registerEnergyFunctionWithName(this,"MolecularContact");
   potts->registerCellGChangeWatcher(this);
   simulator->registerSteerableObject(this);
   potts->getCellFactoryGroupPtr()->registerClass(&molecularContactDataAccessor);
   update(_xmlData,true);
   weightDistance = false;


}


void MolecularContactPlugin::initializeMolecularConcentrations(){
   Potts3D *potts = simulator->getPotts();
   CellInventory::cellInventoryIterator cInvItr;
   CellG * cell;

//    BasicClassAccessor<MolecularContactDataContainer> molecularContactDataAccessor;
//    BasicClassAccessor<MolecularContactDataContainer> * molecularContactDataAccessorPtr;

   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){

      cell=cellInventoryPtr->getCell(cInvItr);
//       ContactLocalFlexDataContainer *dataContainer = contactDataContainerAccessor.get(cell->extraAttribPtr);
//       dataContainer->localDefaultContactEnergies = contactEnergyEqnsArray;


      MolecularContactDataContainer *dataContainer = molecularContactDataAccessor.get(cell->extraAttribPtr);
      for(iterMoleName = moleculeNameMap.begin(); iterMoleName != moleculeNameMap.end(); iterMoleName++)
      {
         Log(LOG_DEBUG) << (*iterMoleName).first << " is the molecule and the value is: " << iterMoleName->second;
         dataContainer->moleculeNameMapContainer[iterMoleName->first] = iterMoleName->second;
      }
      Log(LOG_DEBUG) << " Cell id: " << cell->id;
   }
   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){

      cell=cellInventoryPtr->getCell(cInvItr);
      MolecularContactDataContainer *dataContainer = molecularContactDataAccessor.get(cell->extraAttribPtr);
      for(iterMoleName = dataContainer->moleculeNameMapContainer.begin(); iterMoleName != dataContainer->moleculeNameMapContainer.end(); iterMoleName++)
      {
         Log(LOG_DEBUG) << "From Cell: " << cell->id;
         Log(LOG_DEBUG) << (*iterMoleName).first << " is the molecule and the value is: " << iterMoleName->second;
      }
   }

//    exit(0);
}



void MolecularContactPlugin::extraInit(Simulator *simulator) {
  Potts3D *potts = simulator->getPotts();
  cellFieldG = potts->getCellFieldG();

  fieldDim=cellFieldG ->getDim();
  boundaryStrategy=BoundaryStrategy::getInstance();
  bool pluginAlreadyRegisteredFlag;
}

void MolecularContactPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell){
   if(!initialized && sim->getStep()==0){
      initializeMolecularConcentrations();
      initialized=true;
//       exit(0);
   }
//    initializeMolecularConcentrations();
//    exit(0);
}

double MolecularContactPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

   double energy = 0;
   unsigned int token = 0;
   double distance = 0;
   Point3D n;

   CellG *nCell=0;
   WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
   Neighbor neighbor;

   if(newCell) {
      Log(LOG_DEBUG) <<  "newCell type: " << (int)newCell->type;
   }else{
      Log(LOG_DEBUG) << "newCell type: " << "medium";
   }

   if(oldCell) {
      Log(LOG_DEBUG) << "oldCell type: " << (int)oldCell->type;
   }else{
      Log(LOG_DEBUG) << "oldCell type: " << "medium";
   }

   if(weightDistance){
      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
         if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
            continue;
         }
         nCell = fieldG->get(neighbor.pt);
         if(nCell!=oldCell){
            energy -= contactEnergyEqn(oldCell, nCell) / neighbor.distance;
         }
         if(nCell!=newCell){
            energy += contactEnergyEqn(newCell, nCell) / neighbor.distance;
         }


      }
   }else{
      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
         if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
            continue;
         }

         nCell = fieldG->get(neighbor.pt);
         if(nCell!=oldCell){
            double tempEnergy = contactEnergyEqn(oldCell, nCell);
            Log(LOG_DEBUG) <<  "EnergyBEN3: " << tempEnergy<< "\n";
            Log(LOG_DEBUG) << "Before energy: " << energy;
            energy -= tempEnergy;
            Log(LOG_DEBUG) << "After energy: " << energy;
         }
         if(nCell!=newCell){
            double tempEnergy = contactEnergyEqn(newCell, nCell);
            Log(LOG_DEBUG) << "EnergyBEN4: " << tempEnergy << "\n";
            Log(LOG_DEBUG) << "Before energy: " << energy;
            energy += tempEnergy;
            Log(LOG_DEBUG) << "After energy: " << energy;
         }


      }


   }
   Log(LOG_DEBUG) << "pt="<<pt<<" energy="<<energy;
   Log(LOG_DEBUG) << "\n\n";
//    sleep(2);
   return energy;


}
void MolecularContactPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
   automaton = potts->getAutomaton();
   moleculeNameMap.clear();
   CC3DXMLElementList cellTypeVec=_xmlData->getElements("Molecule");



   for (int i = 0 ; i<cellTypeVec.size(); ++i){
      moleculeNameMap[cellTypeVec[i]->getAttribute("Name")]=cellTypeVec[i]->getAttributeAsByte("Value");
   }

   for(iterMoleName = moleculeNameMap.begin(); iterMoleName != moleculeNameMap.end(); iterMoleName++)
   {
      Log(LOG_DEBUG) << (*iterMoleName).first << " is the molecule and the value is: " << iterMoleName->second;
   }

   CC3DXMLElementList energyEqnVec=_xmlData->getElements("EneryEqn");
   string temp;
   set<unsigned char> cellTypesSet;

   for (int i = 0 ; i<energyEqnVec.size(); ++i){
//       temp =
      Log(LOG_DEBUG) << "Energy Equation Matrix: ";
      Log(LOG_DEBUG) << "Got Here Assignment Start: " << energyEqnVec.size() << " : " << i << "\n";
      setContactEnergyEqn(energyEqnVec[i]->getAttribute("Type1"), energyEqnVec[i]->getAttribute("Type2"), energyEqnVec[i]->getText());

//       //inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
      cellTypesSet.insert(automaton->getTypeId(energyEqnVec[i]->getAttribute("Type1")));

      cellTypesSet.insert(automaton->getTypeId(energyEqnVec[i]->getAttribute("Type2")));

      temp = energyEqnVec[i]->getText();
      Log(LOG_DEBUG) << temp;

   }
   Log(LOG_DEBUG) << "Got Here Assignment Start\n";

   //Now that we know all the types used in the simulation we will find size of the contactEnergyEqnsArray
   vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector

   int size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
   size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated

   int index ;
   contactEnergyEqnsArray.clear();
   contactEnergyEqnsArray.assign(size,vector<string>(size,"empty"));

   for(int i = 0 ; i < size ; ++i)
      for(int j = 0 ; j < size ; ++j){

      index = getIndex(cellTypesVector[i],cellTypesVector[j]);

      contactEnergyEqnsArray[i][j] = contactEnergyEqns[index];

      }
      Log(LOG_DEBUG) << "size="<<size;
      for(int i = 0 ; i < size ; ++i)
         for(int j = 0 ; j < size ; ++j){
         Log(LOG_DEBUG) << "contact["<<i<<"]["<<j<<"]="<<contactEnergyEqnsArray[i][j];

         }

			//Here I initialize max neighbor index for direct acces to the list of neighbors
         boundaryStrategy=BoundaryStrategy::getInstance();
         maxNeighborIndex=0;

         if(_xmlData->getFirstElement("Depth")){
            maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
         }else{
            if(_xmlData->getFirstElement("NeighborOrder")){

               maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());
            }else{
               maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

            }

         }
         Log(LOG_DEBUG) << "Contact maxNeighborIndex="<<maxNeighborIndex;
         Log(LOG_DEBUG) << "Got Here End\n";
}


void MolecularContactPlugin::setContactEnergyEqn(const string typeName1,const string typeName2,const string energyEqn) {


   char type1 = automaton->getTypeId(typeName1);
   char type2 = automaton->getTypeId(typeName2);

   int index = getIndex(type1, type2);

   contactEnergyEqns_t::iterator it = contactEnergyEqns.find(index);
   ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
         " already set!", it == contactEnergyEqns.end());

   contactEnergyEqns[index] = energyEqn;
}


void MolecularContactPlugin::setContactEnergy(const string typeName1,const string typeName2,const double energy) {

   /*
   char type1 = automaton->getTypeId(typeName1);
   char type2 = automaton->getTypeId(typeName2);

   int index = getIndex(type1, type2);

   contactEnergyEqns_t::iterator it = contactEnergyEqns.find(index);
   ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
         " already set!", it == contactEnergyEqns.end());

   contactEnergyEqns[index] = energyEqn;*/

}

int MolecularContactPlugin::getIndex(const int type1, const int type2) const {
   if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
   else return ((type2 + 1) | ((type1 + 1) << 16));
}

std::string MolecularContactPlugin::toString(){
   return "MolecularContact";
}

// void ContactLocalProductPlugin::setJVecValue(CellG * _cell, unsigned int _index,float _value){
//    (contactProductDataAccessor.get(_cell->extraAttribPtr)->jVec)[_index]=_value;
// }


void MolecularContactPlugin::setConcentration(CellG * _cell,std::string MoleName,float val) {
   MolecularContactDataContainer *dataContainer = molecularContactDataAccessor.get(_cell->extraAttribPtr);

   Log(LOG_DEBUG) << "Before\n";

   for(iterMoleName = dataContainer->moleculeNameMapContainer.begin(); iterMoleName != dataContainer->moleculeNameMapContainer.end(); iterMoleName++)
   {
      Log(LOG_DEBUG) << "From Cell: " << _cell->id;
      Log(LOG_DEBUG) << (*iterMoleName).first << " is the molecule and the value is: " << iterMoleName->second;
   }


   molecularContactDataAccessor.get(_cell->extraAttribPtr)->moleculeNameMapContainer[MoleName]=val;
   Log(LOG_DEBUG) << "After\n";
   for(iterMoleName = dataContainer->moleculeNameMapContainer.begin(); iterMoleName != dataContainer->moleculeNameMapContainer.end(); iterMoleName++)
   {
      Log(LOG_DEBUG) << "From Cell: " << _cell->id;
      Log(LOG_DEBUG) << (*iterMoleName).first << " is the molecule and the value is: " << iterMoleName->second;
   }
//    sleep(2);
//    sleep(1);
}


double MolecularContactPlugin::contactEnergyEqn(const CellG *cell1, const CellG *cell2) {

   int tempMCS = sim->getStep();
   Log(LOG_DEBUG) << "Current Step: " << tempMCS<< "\n";
   Log(LOG_DEBUG) << "mcsState: " << mcsState<< "\n";
   Log(LOG_DEBUG) << " all equations\n";
   int size = contactEnergyEqnsArray.size();
   for(int i = 0 ; i < size ; ++i){
      for(int j = 0 ; j < size ; ++j){
//          contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0]

      }
   }

//Dont' know what this is for!!!
   if ((mcsState == -1)){
      Log(LOG_DEBUG) << "Here1\n";
      Log(LOG_DEBUG) << "I'm sleepy Return array value: " << contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
       return contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
       //sleep(2);
   }

   else {
      mcsState = sim->getStep();

      CellInventory::cellInventoryIterator cInvItr;
      CellG * cell;
      /*
      alpha = molecularContactDataAccessor.get(_cell->extraAttribPtr)->moleculeNameMapContainer[MoleName];
      beta = molecularContactDataAccessor.get(_cell->extraAttribPtr)->moleculeNameMapContainer[MoleName];
      gamma = molecularContactDataAccessor.get(_cell->extraAttribPtr)->moleculeNameMapContainer[MoleName];


      val = func(alpha,beta,gamma);
      return val;*/

      if(cell1){
         Log(LOG_DEBUG) <<  "New Cell: " << cell1->id << " Type: " << (int)cell1->type;
         MolecularContactDataContainer *dataContainer = molecularContactDataAccessor.get(cell1->extraAttribPtr);
         for(iterMoleName = dataContainer->moleculeNameMapContainer.begin(); iterMoleName != dataContainer->moleculeNameMapContainer.end(); iterMoleName++)
         {
         }
      }
      if(cell2) {
         Log(LOG_DEBUG) <<  "Old Cell: " << cell2->id << " Type: " << (int)cell2->type;
         MolecularContactDataContainer *dataContainer = molecularContactDataAccessor.get(cell2->extraAttribPtr);
         for(iterMoleName = dataContainer->moleculeNameMapContainer.begin(); iterMoleName != dataContainer->moleculeNameMapContainer.end(); iterMoleName++)
         {
         }
      }

   //    return contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];

      /*cell = const_cast<CellG *>(cell1);
      if(cell){
         if((int)cell->id == 146) {
            string temp = "X";
//             setConcentration(cell,temp,3.14159);
         }
      }*/


      Py_ssize_t pos = 0;
      float fval;
      string skey;
      string eqn = contactEnergyEqnsArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
      int flag = 0;

      PyObject *key, *value;
      PyObject* nameDictPY = PyDict_New();
      PyObject * val = PyFloat_FromDouble(5);

   //    PyDict_SetItemString(nameDictPY, molecule, val);
      string temp;
      if(cell1){
         MolecularContactDataContainer *dataContainer = molecularContactDataAccessor.get(cell1->extraAttribPtr);
         for(iterMoleName = dataContainer->moleculeNameMapContainer.begin(); iterMoleName != dataContainer->moleculeNameMapContainer.end(); iterMoleName++) {
            temp = (*iterMoleName).first;
            temp.append("_1");
            val = PyFloat_FromDouble(iterMoleName->second);
            PyDict_SetItemString(nameDictPY, temp.c_str(), val);
            flag++;
         }
      }

      if(cell2){
         MolecularContactDataContainer *dataContainer = molecularContactDataAccessor.get(cell2->extraAttribPtr);
         for(iterMoleName = dataContainer->moleculeNameMapContainer.begin(); iterMoleName != dataContainer->moleculeNameMapContainer.end(); iterMoleName++) {
            temp = (*iterMoleName).first;
            val = PyFloat_FromDouble(iterMoleName->second);
            temp.append("_2");
            PyDict_SetItemString(nameDictPY, temp.c_str(), val);
            flag++;
         }
      }

      pos = 0;
      while (PyDict_Next(nameDictPY, &pos, &key, &value)) {
         Log(LOG_DEBUG) << "In dictionary\n";
         skey = PyString_AsString(key);
         fval = (float)PyFloat_AsDouble(value);
         Log(LOG_DEBUG) <<  "\t" << skey << ": " << fval;
      }
   //    PyObject* po_main = PyImport_AddModule("__main__");


      PyGILState_STATE gstate;
      gstate = PyGILState_Ensure();

      /* Perform Python actions here.  */
      PyObject* po_main = PyImport_AddModule("__main__");
//       PyRun_SimpleString("from time import time,ctime\n"
//             "print 'Today is',ctime(time())\n");

      /* evaluate result */

      /* Release the thread. No Python API allowed beyond this point. */

      PyRun_SimpleString("energy = 0.0");


      PyObject_SetAttrString(po_main,"pydict",nameDictPY);


//       string code ("count =0; print energy; energy = 0;print pydict;\nfor i in pydict:\n\tcount+=1\n\tprint i,\n\tvars()[i] = pydict[i]\n\tprint pydict[i]\n\tprint '  Count: ', count\n\tif count == len(pydict):\n\t\tenergy =  ");


      string code ("count =0; energy = 0;\nfor i in pydict:\n\tcount+=1\n\tvars()[i] = pydict[i]\n\tif count == len(pydict):\n\t\tenergy =  ");

//       string code ("count =0; energy = 0;\nfor i in pydict:\n\tcount+=1\n\tvars()[i] = pydict[i]\n");


      PyRun_SimpleString("from math import *");
      code.append(eqn);
//       code.append("\n\t\tprint 'python energy: ', energy");
      Log(LOG_DEBUG) << "Eqn: " << eqn;
      Log(LOG_DEBUG) << "Eqn.cstr(): " << eqn.c_str();
      Log(LOG_DEBUG) << "code: " << code;
      Log(LOG_DEBUG) << "code.cstr(): " << code.c_str();

      PyRun_SimpleString(code.c_str());
      //PyRun_SimpleString(eqn.c_str());
//       PyRun_SimpleString("energy  = energy +0.1");
      PyObject* po_energy = PyObject_GetAttrString(po_main, "energy");

      double po_evalue = PyFloat_AsDouble(po_energy);
      Log(LOG_DEBUG) << "C++ Check Float " << PyFloat_Check(po_energy);
      Log(LOG_DEBUG) << "C++ Check Float po_evalue" << po_evalue;
      PyGILState_Release(gstate);
      Log(LOG_DEBUG) << "Release Gil State\n";
      //    sleep(2);
      Py_XDECREF(nameDictPY);

      //    Py_XDECREF(val);
   //    Py_XDECREF(key);
   //    Py_XDECREF(value);
   //    Py_XDECREF(po_main);
      return po_evalue;
   }
}

std::string MolecularContactPlugin::steerableName(){
   return toString();
}
