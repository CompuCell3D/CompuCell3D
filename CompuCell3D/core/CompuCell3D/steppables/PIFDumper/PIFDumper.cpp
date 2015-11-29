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

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/plugins/CellType/CellTypePlugin.h>

using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>


// // // #include <string>
// // // #include <sstream>
// // // #include <iostream>
// // // #include <map>
using namespace std;


#include "PIFDumper.h"

PIFDumper::PIFDumper() :
  potts(0),pifFileExtension("pif") {}

PIFDumper::PIFDumper(string filename) :
  potts(0),pifFileExtension("pif") {}

void PIFDumper::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	
   potts = simulator->getPotts();
   
   ostringstream numStream;
   string numString;
   
   numStream<<simulator->getNumSteps();;
   
   numString=numStream.str();
   
   numDigits=numString.size();
   typePlug = (CellTypePlugin*)(Simulator::pluginManager.get("CellType"));

   simulator->registerSteerableObject(this);

	update(_xmlData,true);
}

void PIFDumper::step( const unsigned int currentStep){
   
   ostringstream fullNameStr;
   fullNameStr<<pifname;
   fullNameStr.width(numDigits);
   fullNameStr.fill('0');
   fullNameStr<<currentStep<<"."<<pifFileExtension;
   

   ofstream pif(fullNameStr.str().c_str());

   WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
   Dim3D dim = cellFieldG->getDim();
   Point3D pt;
   CellG * cell;

   for (int x = 0 ; x < dim.x ; ++x)
      for (int y = 0 ; y < dim.y ; ++y)
         for (int z = 0 ; z < dim.z ; ++z){
            pt.x=x;
            pt.y=y;
            pt.z=z;
            cell=cellFieldG->get(pt);
            if(cell){
               pif<<cell->id<<"\t";
               pif<<typePlug->getTypeName(cell->type)<<"\t";
               pif<<pt.x<<"\t"<<pt.x<<"\t";
               pif<<pt.y<<"\t"<<pt.y<<"\t";
               pif<<pt.z<<"\t"<<pt.z<<"\t";
               pif<<endl;
            }
         }
}



void PIFDumper::start() {


} 


void PIFDumper::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

   // Frequency is handled at a higher level, no need to handle it here.
	pifname=_xmlData->getFirstElement("PIFName")->getText();
	if(_xmlData->findElement("PIFFileExtension"))
		pifFileExtension=_xmlData->getFirstElement("PIFFileExtension")->getText();

}

std::string PIFDumper::toString(){
   return "PIFDumper";

}


std::string PIFDumper::steerableName(){
   return toString();

}

