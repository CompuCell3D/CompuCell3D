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

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;

// // // #include <iostream>
// // // #include <string>
// // // #include <algorithm>
// // // #include <PublicUtilities/Units/Unit.h>
using namespace std;

#include "TemplatePlugin.h"

// // // #include <PublicUtilities/StringUtils.h>

#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>
// // // #include <CompuCell3D/ClassRegistry.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>


// TemplatePlugin::TemplatePlugin() : potts(0) {}

TemplatePlugin::~TemplatePlugin() {}

void TemplatePlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData){
	potts=simulator->getPotts();
	xmlData=_xmlData;
	sim=simulator;
	sim->getPotts()->registerEnergyFunctionWithName(this,toString());
	sim->registerSteerableObject(this);
	TemplateChemicalFieldName = "FGF-TEST";
	TemplateChemicalFieldSource="FlexibleDiffusionSolverFE";
}

void TemplatePlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
        cerr << "GoT here\n";
        string temp =_xmlData->getFirstElement("TemplateString")->getText();
        cerr << "TemplateString: " << temp << endl;

        vector<string> TemplateStringVector;

        //void parseStringIntoList(std::string &str,std::vector<std::string> &strVec,std::string separator)
        //#include <PublicUtilities/StringUtils.h>
        temp =_xmlData->getFirstElement("TemplateStrings")->getText();
        parseStringIntoList(temp , TemplateStringVector , ",");
        for(int i = 0; i < TemplateStringVector.size(); i++) {
            cerr<<"TemplateStringVector[i]="<<TemplateStringVector[i]<<endl;
        }

        CC3DXMLElementList energyVec=_xmlData->getElements("TemplateEnergy");

        for(int i = 0; i < energyVec.size(); i++) {
            cerr <<"Type 1: " << energyVec[i]->getAttribute("Type1") << " Type 2: " << energyVec[i]->getAttribute("Type2")
            << " Value: " << energyVec[i]->getDouble() << endl;

        }

        CC3DXMLElementList cellTypeVec=_xmlData->getElements("TemplateType");



        //<TemplateType TypeName="omega" TypeId="59"/>
        for(int i = 0; i < cellTypeVec.size(); i++) {
            cerr << "TypeName: " << cellTypeVec[i]->getAttribute("TypeName") << " TypeId: " << (int)cellTypeVec[i]->getAttributeAsByte("TypeId") << endl;
        }


        CC3DXMLElementList chemicalFieldXMlList=_xmlData->getElements("TemplateChemicalField");

        for(int i= 0; i < chemicalFieldXMlList.size(); i++) {
            TemplateChemicalFieldName = chemicalFieldXMlList[i]->getAttribute("Name");
            TemplateChemicalFieldSource = chemicalFieldXMlList[i]->getAttribute("Source");
            cerr <<"Source: " << chemicalFieldXMlList[i]->getAttribute("Source") << " Name: " << chemicalFieldXMlList[i]->getAttribute("Name") << endl;

            CC3DXMLElementList chemotactByTypeXMlList=chemicalFieldXMlList[i]->getElements("TemplateaChemotaxisByType");
            for(int j= 0; j < chemotactByTypeXMlList.size(); j++) {
                cerr <<"Type: " << chemotactByTypeXMlList[j]->getAttribute("Type")<< " Name: " << chemicalFieldXMlList[i]->getAttribute("Name") << endl;
                if(chemotactByTypeXMlList[j]->findAttribute("Lambda")){
                    cerr << " Lambda: " << chemotactByTypeXMlList[j]->getAttributeAsDouble("Lambda");
                }
                if(chemotactByTypeXMlList[j]->findAttribute("SaturationCoef")){
                    cerr << " SaturationCoef: " << chemotactByTypeXMlList[j]->getAttributeAsDouble("SaturationCoef");
                }
                cerr << "\n";
            }
        }


        ClassRegistry *classRegistry=sim->getClassRegistry();
        Steppable * steppable;
        fieldVec.clear();
        cerr << "Finished Setting Up\n";


        fieldVec.assign(1.0,0);//allocate fieldVec

        map<string,Field3D<float>*> & nameFieldMap = sim->getConcentrationFieldNameMap();
        map<string,Field3D<float>*>::iterator mitr=nameFieldMap.find(TemplateChemicalFieldName);

        if(mitr!=nameFieldMap.end()){
            fieldVec[0]=mitr->second;
            
        }else{
            ASSERT_OR_THROW("No chemical field has been loaded!", fieldVec[0]);

        }        
        //OLD STYLE
        // steppable=classRegistry->getStepper(TemplateChemicalFieldSource);        
        // fieldVec[0]=((DiffusableVector<float> *) steppable)->getConcentrationField(TemplateChemicalFieldName);



}


void TemplatePlugin::extraInit(Simulator *simulator){
	update(xmlData);
}






double TemplatePlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	/// E = lambda * (volume - targetTemplate) ^ 2

    cerr << "pt.x: " << pt.x << " pt.y: " << pt.y << " pt.z: " << pt.z << endl;
    if(newCell) {
        cerr << "newCell id: " << newCell->id << endl;
    }
    if(oldCell) {
        cerr << "newCell id: " << oldCell->id << endl;
    }

    for(unsigned int i = 0 ; i < fieldVec.size() ; ++i){
        cerr << "Concentration: " << fieldVec[i]->get(pt)  << endl;
        if(fieldVec[i]->get(pt) > 1) {
            cerr << "\t NON ZERO VALUE!!!\n";
        }
    }





    //exit(0);
	return 0;

}


std::string TemplatePlugin::steerableName(){
	return pluginName;
	//return "Template";
}

std::string TemplatePlugin::toString(){
	return pluginName;
	//return "Template";
}




