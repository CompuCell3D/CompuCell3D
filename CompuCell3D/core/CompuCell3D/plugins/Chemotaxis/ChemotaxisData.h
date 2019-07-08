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

#ifndef CHEMOTAXISDATA_H
#define CHEMOTAXISDATA_H
 #include <CompuCell3D/CC3D.h>
 
// // // #include <string>
// // // #include <vector>
// // // #include <map>
// // // #include <iostream>

// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <BasicUtils/BasicException.h>
// // // #include <PublicUtilities/StringUtils.h>
#include "ChemotaxisDLLSpecifier.h"
// // // #include <CompuCell3D/Automaton/Automaton.h>

namespace CompuCell3D {
	class ChemotaxisPlugin;
	
	
   class CHEMOTAXIS_EXPORT  ChemotaxisData{
      public:
         ChemotaxisData(float _lambda=0.0 , float _saturationCoef=0.0 , std::string _typeName=""):
         lambda(_lambda),saturationCoef(_saturationCoef),typeName(_typeName),formulaPtr(0),formulaName("SimpleChemotaxisFormula"),
	     chemotaxisFormulaDictPtr(0),
		 automaton(0),
		 allowChemotaxisBetweenCompartmentsGlobal(true)
         {}
	
         float lambda;
         float saturationCoef;
		 float powerLevel;
         std::string formulaName;
         typedef float (ChemotaxisPlugin::*chemotaxisEnergyFormulaFcnPtr_t)(float,float,ChemotaxisData &);
         chemotaxisEnergyFormulaFcnPtr_t  formulaPtr;
		 std::map<std::string,chemotaxisEnergyFormulaFcnPtr_t> *chemotaxisFormulaDictPtr;
		 
		 bool allowChemotaxisBetweenCompartmentsGlobal;

         std::string typeName;
         std::vector<unsigned char> chemotactTowardsTypesVec;
         std::string chemotactTowardsTypesString;
         Automaton * automaton;

         void setLambda(float _lambda){lambda=_lambda;}
         float getLambda(){return lambda;}
         void setType(std::string _typeName){typeName=_typeName;}
         void setChemotactTowards(std::string _chemotactTowardsTypesString){
            chemotactTowardsTypesString=_chemotactTowardsTypesString;
         }
		 
		 void setChemotaxisFormulaByName(std::string _formulaName){
			formulaName=_formulaName;
			if (chemotaxisFormulaDictPtr){
				std::map<std::string,chemotaxisEnergyFormulaFcnPtr_t>::iterator mitr;
				mitr=chemotaxisFormulaDictPtr->find(_formulaName);
				if(mitr!=chemotaxisFormulaDictPtr->end()){
					formulaName=_formulaName;
					formulaPtr=mitr->second;
				}
			}


		 }

		 void initializeChemotactTowardsVectorTypes(std::string _chemotactTowardsTypesString){
			chemotactTowardsTypesVec.clear();
			std::vector<std::string> vecTypeNamesTmp;
			parseStringIntoList(_chemotactTowardsTypesString,vecTypeNamesTmp,",");

			for(int idx=0 ; idx < vecTypeNamesTmp.size() ; ++idx){

				unsigned char typeTmp=automaton->getTypeId(vecTypeNamesTmp[idx]);
				chemotactTowardsTypesVec.push_back(typeTmp);
			}
		 }

		 void assignChemotactTowardsVectorTypes(std::vector<int> _chemotactTowardsTypesVec){
			chemotactTowardsTypesVec.clear();
			for(int idx=0 ; idx < _chemotactTowardsTypesVec.size() ; ++idx){				
				chemotactTowardsTypesVec.push_back(_chemotactTowardsTypesVec[idx]);
			}
		 }

		 std::vector<int> getChemotactTowardsVectorTypes(){
			std::vector<int> chemotactTowardsTypesVecInt;
			
			for(int idx=0 ; idx < chemotactTowardsTypesVec.size() ; ++idx){				
				chemotactTowardsTypesVecInt.push_back(chemotactTowardsTypesVec[idx]);
			}
			return chemotactTowardsTypesVecInt;
		 }

         void setSaturationCoef(float _saturationCoef){
            saturationCoef=_saturationCoef;
            
			if (chemotaxisFormulaDictPtr){
				std::map<std::string,chemotaxisEnergyFormulaFcnPtr_t>::iterator mitr;
				mitr=chemotaxisFormulaDictPtr->find("SaturationChemotaxisFormula");
				if(mitr!=chemotaxisFormulaDictPtr->end()){
					formulaName="SaturationChemotaxisFormula";
					formulaPtr=mitr->second;					
				}
			}

         }
         void setSaturationLinearCoef(float _saturationCoef){
            saturationCoef=_saturationCoef;
            
			if (chemotaxisFormulaDictPtr){
				std::map<std::string,chemotaxisEnergyFormulaFcnPtr_t>::iterator mitr;
				mitr=chemotaxisFormulaDictPtr->find("SaturationLinearChemotaxisFormula");
				if(mitr!=chemotaxisFormulaDictPtr->end()){
					formulaName="SaturationLinearChemotaxisFormula";
					formulaPtr=mitr->second;
				}
			}

         }

         void outScr(){
            using namespace std;
            cerr<<"**************ChemotaxisData**************"<<endl;
            cerr<<"formulaPtr="<<formulaPtr<<endl;
            cerr<<"lambda="<<lambda<<" saturationCoef="<<saturationCoef<<" typaName="<<typeName<<endl;
            cerr<<"chemotactTowards="<<chemotactTowardsTypesString<<endl;
            cerr<<"Chemotact towards types:"<<endl;
            for (int i = 0 ; i < chemotactTowardsTypesVec.size() ; ++i){
               cerr<<"chemotact Towards type id="<<(int)chemotactTowardsTypesVec[i]<<endl;
            }
            cerr<<"**************ChemotaxisData END**************"<<endl;
         }

         bool okToChemotact( const CellG * _oldCell, const CellG * _newCell){

			 if (!this->allowChemotaxisBetweenCompartmentsGlobal) {
				 if (_oldCell && _newCell && (_newCell->clusterId == _oldCell->clusterId)) {
					 return false;
				 }	
				  
			 }

            if(chemotactTowardsTypesVec.size()==0){ //chemotaxis always enabled for this cell
               return true;
            }
            //will chemotact towards only specified cell types
         
            unsigned char type= (_oldCell?_oldCell->type:0);
         
            for(unsigned int i = 0 ; i < chemotactTowardsTypesVec.size() ; ++i){
               if (type==chemotactTowardsTypesVec[i])
                  return true;
            }
         
            return false;
         }
   


   };

   class  ChemotaxisFieldData{
      public:
         ChemotaxisFieldData()
         {}
         std::string chemicalFieldSource;//this is deprecated. have to check if used in python wrapper 
         std::string chemicalFieldName;
         std::vector<ChemotaxisData> vecChemotaxisData;
         ChemotaxisData *ChemotaxisByType(){
            vecChemotaxisData.push_back(ChemotaxisData());
            return & vecChemotaxisData[vecChemotaxisData.size() -1];
         }
         void Source(std::string _chemicalFieldSource){
            chemicalFieldSource=_chemicalFieldSource;
         }
         void Name(std::string _chemicalFieldName){chemicalFieldName=_chemicalFieldName;}
         ChemotaxisData * getChemotaxisData(int _index){
            if(_index<0 || _index>vecChemotaxisData.size()-1)
               return 0;
            return &vecChemotaxisData[_index];
         }
   };
  
};

#endif
