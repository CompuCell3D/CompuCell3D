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


// #include <map>
// #include <vector>
// #include <CompuCell3D/Automaton/CellType.h>
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;




// #include <CompuCell3D/Simulator.h>
//  #include <CompuCell3D/Potts3D/Potts3D.h>

//#include <XMLCereal/XMLPullParser.h>
//#include <XMLCereal/XMLSerializer.h>

// #include <BasicUtils/BasicString.h>
// #include <BasicUtils/BasicException.h>

// #include <XMLUtils/CC3DXMLElement.h>

#include <iostream>
using namespace std;


#include "CellTypePlugin.h"
//#include "CellTypeParseData.h"



std::string CellTypePlugin::toString(){
   return "CellType";
}

std::string CellTypePlugin::steerableName(){
  return toString();
}

CellTypePlugin::CellTypePlugin() {classType = new CellType();}

CellTypePlugin::~CellTypePlugin() {
   if (classType){
      delete classType;
      classType=0;
   }
}

void CellTypePlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData){

   potts = simulator->getPotts();
   potts->registerCellGChangeWatcher(this);
   potts->registerAutomaton(this);
   update(_xmlData);
   simulator->registerSteerableObject((SteerableObject*)this);

}



void CellTypePlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
  
      typeNameMap.clear();
      nameTypeMap.clear();
      vector<unsigned char> frozenTypeVec;
		CC3DXMLElementList cellTypeVec=_xmlData->getElements("CellType");

		for (int i = 0 ; i<cellTypeVec.size(); ++i){
			typeNameMap[cellTypeVec[i]->getAttributeAsByte("TypeId")]=cellTypeVec[i]->getAttribute("TypeName");
			nameTypeMap[cellTypeVec[i]->getAttribute("TypeName")]=cellTypeVec[i]->getAttributeAsByte("TypeId");

			if(cellTypeVec[i]->findAttribute("Freeze"))
				frozenTypeVec.push_back(cellTypeVec[i]->getAttributeAsByte("TypeId"));
		}

      potts->setFrozenTypeVector(frozenTypeVec);
      
  

}

void CellTypePlugin::init(Simulator *simulator, ParseData * _pd) {
	//cerr<<"CELL TYPE THIS IS _pd="<<_pd<<endl;
 //  potts = simulator->getPotts();
 //  potts->registerCellGChangeWatcher(this);
 //  potts->registerAutomaton(this);
 //  update(_pd);
 //  simulator->registerSteerableObject((SteerableObject*)this);

}

void CellTypePlugin::update(ParseData *_pd, bool _fullInitFlag){

//   if(_pd){
//      typeNameMap.clear();
//      nameTypeMap.clear();
//      CellTypeParseData * cpdPtr=(CellTypeParseData *)_pd;
//      vector<unsigned char> frozenTypeVec;
////       cerr<<" cpdPtr->cellTypeTuppleVec.size()="<<cpdPtr->cellTypeTuppleVec.size()<<endl;
//      for(int i = 0 ; i < cpdPtr->cellTypeTuppleVec.size() ; ++i){
//         typeNameMap[cpdPtr->cellTypeTuppleVec[i].typeId]=cpdPtr->cellTypeTuppleVec[i].typeName;
//         nameTypeMap[cpdPtr->cellTypeTuppleVec[i].typeName]=cpdPtr->cellTypeTuppleVec[i].typeId;
//         if(cpdPtr->cellTypeTuppleVec[i].freeze){
//            frozenTypeVec.push_back(cpdPtr->cellTypeTuppleVec[i].typeId);
//         }
//      }
//      potts->setFrozenTypeVector(frozenTypeVec);
//   }

}



unsigned char CellTypePlugin::getCellType(const CellG *cell) const {

   if(!cell) return 0;

   return cell->type;

}


string CellTypePlugin::getTypeName(const char type) const {


  std::map<unsigned char,std::string>::const_iterator typeNameMapItr=typeNameMap.find((const unsigned char)type);

  
  if(typeNameMapItr!=typeNameMap.end()){
      return typeNameMapItr->second;
  }else{
      THROW(string("getTypeName: Unknown cell type  ") + BasicString(type) + "!");
  }


}

unsigned char CellTypePlugin::getTypeId(const string typeName) const {


  std::map<std::string,unsigned char>::const_iterator nameTypeMapItr=nameTypeMap.find(typeName);
  
  
  if(nameTypeMapItr!=nameTypeMap.end()){
      return nameTypeMapItr->second;
  }else{
      THROW(string("getTypeName: Unknown cell type  ") + typeName + "!");
  }

}


unsigned char CellTypePlugin::getMaxTypeId() const {
	cerr<<"typeNameMap.size()="<<typeNameMap.size()<<endl;
	if (! typeNameMap.size()){
		return 0;
	}else{
		return (--(typeNameMap.end()))->first; //returning last type number (unsigned char) 
	}
}

//void CellTypePlugin::readXML(XMLPullParser &in) {
//   
//  pd=&cpd;
//  in.skip(TEXT);
//
//  while (in.check(START_ELEMENT)) {
//    if (in.getName() == "CellType") {
//      CellTypeTupple cellTypeTupple;
//      cellTypeTupple.typeName = in.getAttribute("TypeName").value;
//      cellTypeTupple.typeId = BasicString::parseUByte(in.getAttribute("TypeId").value);
//      cerr<<"typeName="<<cellTypeTupple.typeName<<endl;
//      cerr<<"typeId="<<(short)cellTypeTupple.typeId<<endl;
//  
//      ///deciding whether to freeze or not
//      int attrNum=in.findAttribute("Freeze");
//      if(attrNum != -1){
//         cellTypeTupple.freeze=true;
//      }
//      in.matchSimple();
//      cpd.cellTypeTuppleVec.push_back(cellTypeTupple);
//    }
//    else {
//      throw BasicException(string("Unexpected element '") + in.getName() +
//                           "'!", in.getLocation());
//    }
//
//    in.skip(TEXT);
//  }
//
//  
//}
//
//void CellTypePlugin::writeXML(XMLSerializer &out) {
//}
//
