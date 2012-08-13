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

#ifndef PARSER_H
#define PARSER_H

// #include <CompuCell3D/dllDeclarationSpecifier.h>
#include <CompuCell3D/ParseData.h>
#include <vector>
#include <string>
#include <XMLUtils/CC3DXMLElement.h>

class CC3DXMLElement;

namespace CompuCell3D {

  class  ParserStorage{
      public:

			ParserStorage():pottsCC3DXMLElement(0),updatePottsCC3DXMLElement(0),metadataCC3DXMLElement(0),updateMetadataCC3DXMLElement(0)
			{}
			CC3DXMLElementList steppableCC3DXMLElementVector;
			CC3DXMLElementList pluginCC3DXMLElementVector;
			CC3DXMLElement * pottsCC3DXMLElement;			
			CC3DXMLElement * metadataCC3DXMLElement;			

			CC3DXMLElementList updateSteppableCC3DXMLElementVector;
			CC3DXMLElementList updatePluginCC3DXMLElementVector;
			CC3DXMLElement * updatePottsCC3DXMLElement;
			CC3DXMLElement * updateMetadataCC3DXMLElement;			

			void addPottsDataCC3D(CC3DXMLElement * _element){pottsCC3DXMLElement=_element;}
			void addMetadataDataCC3D(CC3DXMLElement * _element){metadataCC3DXMLElement=_element;}
			void addPluginDataCC3D(CC3DXMLElement * _element){pluginCC3DXMLElementVector.push_back(_element);}
			void addSteppableDataCC3D(CC3DXMLElement * _element){steppableCC3DXMLElementVector.push_back(_element);}	

			



   };


  

};
#endif
