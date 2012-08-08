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
#include <CompuCell3D/Potts3D/CellInventory.h>

using namespace CompuCell3D;

#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include <vector>
#include <string>
#include <stdlib.h>
#include <iostream>

using namespace std;


#include "SimpleArrayPlugin.h"

SimpleArrayPlugin::SimpleArrayPlugin() : Plugin() {scpdPtr=0;}

SimpleArrayPlugin::~SimpleArrayPlugin() {}

void SimpleArrayPlugin::update(ParseData *_pd){

   scpdPtr=(SimpleArrayParseData *)_pd;

   int size;
   
   int index = 0;
   string next;
   string value;


   for(int i = 0; i < scpdPtr->test_string.size(); i++) {
      next = scpdPtr->test_string[i];
      value = next;
      while(next != ",") {
         next = scpdPtr->test_string[i+1];
         value += next;
         i++;
         if(i == scpdPtr->test_string.size()) {
//             cerr << "End of Array: " << value << endl;
            break;
         }
      }
      probMatrix.push_back(BasicString::parseDouble(value));
      value.resize(value.size()-1);
//       cerr << "Hit the comma: " << value << endl;
   }
   
   cerr << "Values: ";
   for(int i = 0; i < probMatrix.size(); i++) {
	   cerr << probMatrix[i] << " ";
   }
   cerr << "\nValues Added to probMatrix\n";
// exit(0);


}

void SimpleArrayPlugin::init(Simulator *_simulator,ParseData *_pd) {
   pd=_pd;
   scpdPtr=(SimpleArrayParseData *)_pd;

   update(scpdPtr);


   Potts3D *potts = _simulator->getPotts();
   
  ///getting cell inventory
   cellInventoryPtr=& potts->getCellInventory();

   


  ///will register SimpleArray here
   BasicClassAccessorBase * simpleArrayAccessorPtr=&simpleArrayAccessor;
   ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
   ///************************************************************************************************  
   potts->getCellFactoryGroupPtr()->registerClass(simpleArrayAccessorPtr);

  
   
   
}

//////
void SimpleArrayPlugin::extraInit(Simulator *_simulator){
   

   Potts3D *potts = _simulator->getPotts();

//     simpleArrayAccessorPtr=new SimpleArrayAccessor<SimpleArray>;
//    potts->getCellFactoryGroupPtr()->registerClass(SimpleArrayAccessorPtr);
    

}
   
void SimpleArrayPlugin::readXML(XMLPullParser &in) {
   pd=&scpd;
   
   
   in.skip(TEXT);
   while (in.check(START_ELEMENT)) {
      if (in.getName() == "Values") {
         cerr << "Probability Matrix Success" << endl;
         scpd.test_string = BasicString::toUpper(in.matchSimple());
      }
      else {
         cerr <<"THROWING BASIC EXPECTION!!!\n";
         throw BasicException(string("Unexpected element '") + in.getName() +
                           "'!", in.getLocation());
      }
      in.skip(TEXT);
   }
   
     
}

void SimpleArrayPlugin::writeXML(XMLSerializer &out) {

}

//void SimpleArrayPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {
   ///Example: how to access probMatrixValues

   ///It will be just comented out
//    if(oldCell){
//       probMatrixTmpPtr=&SimpleArrayAccessor.get(oldCell->extraAttribPtr)->probMatrix;
// 
// 
//    }
// 
//    if(newCell){
// 
//       probMatrixTmpPtr=&SimpleArrayAccessor.get(oldCell->extraAttribPtr)->probMatrix
//    }


//}