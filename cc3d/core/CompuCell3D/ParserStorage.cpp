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


//#include <vector>
//#include <string>
//using namespace std;
//
//
//

//#include "ParserStorage.h"
//
//using namespace CompuCell3D;
//
//
//CC3DXMLElement * ParserStorage::getCC3DModuleData(std::string _moduleType,std::string _moduleName){
//	if(_moduleType=="Potts"){
//		return pottsCC3DXMLElement;
//	}else if(_moduleType=="Plugin"){
//		for (int i = 0 ; i<pluginCC3DXMLElementVector.size() ;  ++i){
//			if (pluginCC3DXMLElementVector[i]->getAttribute("Name")==_moduleName)
//				return pluginCC3DXMLElementVector[i];
//		}
//		return 0;
//	}else if(_moduleType=="Steppable"){
//		for (int i = 0 ; i<pluginCC3DXMLElementVector.size() ;  ++i){
//			if (steppableCC3DXMLElementVector[i]->getAttribute("Type")==_moduleName)
//				return steppableCC3DXMLElementVector[i];
//		}
//		return 0;		
//	}else{
//		return 0;
//	}
//}