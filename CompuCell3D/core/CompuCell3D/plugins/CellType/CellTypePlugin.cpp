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

#include <sstream>

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
#include <iostream>
using namespace std;


#include "CellTypePlugin.h"



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
    
    std::map<unsigned char,std::string>::iterator type_name_mitr;
    std::map<std::string,unsigned char>::iterator name_type_mitr;
    
    vector<unsigned char> frozenTypeVec;
    CC3DXMLElementList cellTypeVec=_xmlData->getElements("CellType");

    for (int i = 0 ; i<cellTypeVec.size(); ++i){
	unsigned char type_id = cellTypeVec[i]->getAttributeAsByte("TypeId");
	std::string type_name = cellTypeVec[i]->getAttribute("TypeName");

	type_name_mitr = typeNameMap.find(type_id);

	ASSERT_OR_THROW("Type id: "+SSTR((int)type_id)+" has already been defined", (type_name_mitr == typeNameMap.end()));
	typeNameMap[type_id] = type_name;

	name_type_mitr = nameTypeMap.find(type_name);

	ASSERT_OR_THROW("Type name "+type_name+" has already been defined",(name_type_mitr == nameTypeMap.end()));
	nameTypeMap[type_name] = type_id;

	if(cellTypeVec[i]->findAttribute("Freeze")){
	    frozenTypeVec.push_back(cellTypeVec[i]->getAttributeAsByte("TypeId"));
	}
    }

      potts->setFrozenTypeVector(frozenTypeVec);
	
      //enforcing the Medium has id =0
	  name_type_mitr = nameTypeMap.find("Medium");
	  if (name_type_mitr == nameTypeMap.end()) {
		  ASSERT_OR_THROW("Medium cell type is not defined. Please define Medium cell type and make sure its type id is set to 0 ",false)
	  }
	  else if (name_type_mitr->second!=0) {
		  ASSERT_OR_THROW("Medium type id can only be set to 0. Please define Medium cell type and make sure its type id is set to 0.",false)
	  }

	  

}

void CellTypePlugin::init(Simulator *simulator, ParseData * _pd) {

}

void CellTypePlugin::update(ParseData *_pd, bool _fullInitFlag){

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


