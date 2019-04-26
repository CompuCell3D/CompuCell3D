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

#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>


using namespace CompuCell3D;

using namespace std;

#include "ChemotaxisData.h"
#include "ChemotaxisPlugin.h"

#include <math.h>


ChemotaxisPlugin::ChemotaxisPlugin():algorithmPtr(&ChemotaxisPlugin::merksChemotaxis),xmlData(0),chemotaxisAlgorithm("merks"),automaton(0) {

	//this dictionary will be used in chemotaxis of individual cells (i.e. in per-cell as opposed to type-based chemotaxis)
	chemotaxisFormulaDict["SaturationChemotaxisFormula"]=&ChemotaxisPlugin::saturationChemotaxisFormula;
	chemotaxisFormulaDict["SaturationLinearChemotaxisFormula"]=&ChemotaxisPlugin::saturationLinearChemotaxisFormula;
	chemotaxisFormulaDict["SimpleChemotaxisFormula"]=&ChemotaxisPlugin::simpleChemotaxisFormula;
    
    //jfg, adding the new formulas here
    chemotaxisFormulaDict["SaturationDifferenceChemotaxisFormula"]=&ChemotaxisPlugin::saturationDifferenceChemotaxisFormula;
    chemotaxisFormulaDict["PowerChemotaxisFormula"]=&ChemotaxisPlugin::powerChemotaxisFormula;
    chemotaxisFormulaDict["Log10DivisionFormula"]=&ChemotaxisPlugin::log10DivisionFormula;
    chemotaxisFormulaDict["LogNatDivisionFormula"]=&ChemotaxisPlugin::logNatDivisionFormula;
    chemotaxisFormulaDict["Log10DifferenceFormula"]=&ChemotaxisPlugin::log10DifferenceFormula;
    chemotaxisFormulaDict["LogNatDifferenceFormula"]=&ChemotaxisPlugin::logNatDifferenceFormula;
	
}

ChemotaxisPlugin::~ChemotaxisPlugin() {
}


void ChemotaxisPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	xmlData=_xmlData;

	sim = simulator;
	potts = simulator->getPotts();

  
  BasicClassAccessorBase * chemotaxisDataAccessorPtr=&chemotaxisDataAccessor;
  ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
  ///************************************************************************************************  
  potts->getCellFactoryGroupPtr()->registerClass(chemotaxisDataAccessorPtr);

	potts->registerEnergyFunctionWithName(this,"Chemotaxis");
	simulator->registerSteerableObject(this);          

}

void ChemotaxisPlugin::extraInit(Simulator *simulator) {

	update(xmlData,true);



}


void ChemotaxisPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	//if(potts->getDisplayUnitsFlag()){
	//	Unit energyUnit=potts->getEnergyUnit();




	//	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
	//	if (!unitsElem){ //add Units element
	//		unitsElem=_xmlData->attachElement("Units");
	//	}

	//	if(unitsElem->getFirstElement("LambdaUnit")){
	//		unitsElem->getFirstElement("LambdaUnit")->updateElementValue(energyUnit.toString());
	//	}else{
	//		CC3DXMLElement * energyElem = unitsElem->attachElement("LambdaUnit",energyUnit.toString());
	//	}
	//}	

	std::vector<ChemotaxisFieldData> chemotaxisFieldDataVec; 


	if(_xmlData->findElement("Algorithm")){
		chemotaxisAlgorithm=_xmlData->getFirstElement("Algorithm")->getText();
		changeToLower(chemotaxisAlgorithm);
	}
	

	//Parsing ChemicalField Sections
	CC3DXMLElementList chemicalFieldXMlList=_xmlData->getElements("ChemicalField");

	//cerr<<"chemicalFieldXMlList.size()="<<chemicalFieldXMlList.size()<<endl;


	for (int i  = 0 ; i < chemicalFieldXMlList.size() ; ++i){




		chemotaxisFieldDataVec.push_back(ChemotaxisFieldData());
		ChemotaxisFieldData & cfd=chemotaxisFieldDataVec[chemotaxisFieldDataVec.size()-1];

		//cfd.chemicalFieldSource = chemicalFieldXMlList[i]->getAttribute("Source");// deprecated
		cfd.chemicalFieldName =chemicalFieldXMlList[i]->getAttribute("Name");

		cfd.vecChemotaxisData.clear();
		//Parsing Chemotaxis by type elements
		CC3DXMLElementList chemotactByTypeXMlList=chemicalFieldXMlList[i]->getElements("ChemotaxisByType");

		//cerr<<"chemotactByTypeXMlList.size()="<<chemotactByTypeXMlList.size()<<endl;

		for (int j = 0 ; j < chemotactByTypeXMlList.size() ; ++j){
			cfd.vecChemotaxisData.push_back(ChemotaxisData());
			ChemotaxisData & cd=cfd.vecChemotaxisData[cfd.vecChemotaxisData.size()-1];
			cd.typeName=chemotactByTypeXMlList[j]->getAttribute("Type");
			
			//jfg, now that there are a bunch of formulas it'd be good to have ifs to select them, instead of relying on 
			// variable names that migh or might not be in the xml
			
			if(chemotactByTypeXMlList[j]->findAttribute("FormulaName"))
			{
				cd.formulaName = chemotactByTypeXMlList[j]->getAttribute("FormulaName");
				
				if(chemotactByTypeXMlList[j]->findAttribute("Lambda")){
					cd.lambda=chemotactByTypeXMlList[j]->getAttributeAsDouble("Lambda");
				}
				
				if(chemotactByTypeXMlList[j]->findAttribute("SaturationCoef"))
				{
					cd.saturationCoef=chemotactByTypeXMlList[j]->getAttributeAsDouble("SaturationCoef");
				}
				else if(chemotactByTypeXMlList[j]->findAttribute("SaturationLinearCoef"))
				{
					cd.saturationCoef=chemotactByTypeXMlList[j]->getAttributeAsDouble("SaturationLinearCoef");
				}
				
				if (chemotactByTypeXMlList[j]->findAttribute("DisallowChemotaxisBetweenCompartments")) 
				{
					cd.allowChemotaxisBetweenCompartmentsGlobal = false;	
				}
				
				if(chemotactByTypeXMlList[j]->findAttribute("ChemotactTowards")){
					//ASSERT_OR_THROW("ChemotactTowards is deprecated now. Please replace it with ChemotactAtInterfaceWith.",chemotaxisFieldDataVec.size());
					cd.chemotactTowardsTypesString=chemotactByTypeXMlList[j]->getAttribute("ChemotactTowards");
				}else if (chemotactByTypeXMlList[j]->findAttribute("ChemotactAtInterfaceWith")){// both keywords are OK
					cd.chemotactTowardsTypesString=chemotactByTypeXMlList[j]->getAttribute("ChemotactAtInterfaceWith");
				}
				
				if( chemotactByTypeXMlList[j]->findAttribute("PowerCoef") )
				{
					cd.powerLevel = chemotactByTypeXMlList[j]->getAttributeAsDouble("PowerCoef");
					if( cd.powerLevel == 1.0 )
					{
						cd.formulaName = false;//powerChemotaxisFormula
					}
				}
				
			}
			else //jfg, end
			{
				if(chemotactByTypeXMlList[j]->findAttribute("Lambda")){
					cd.lambda=chemotactByTypeXMlList[j]->getAttributeAsDouble("Lambda");
				}

				if(chemotactByTypeXMlList[j]->findAttribute("SaturationCoef")){
					cd.saturationCoef=chemotactByTypeXMlList[j]->getAttributeAsDouble("SaturationCoef");
					cd.formulaName="SaturationChemotaxisFormula";
				}

				if (chemotactByTypeXMlList[j]->findAttribute("DisallowChemotaxisBetweenCompartments")) {
					cd.allowChemotaxisBetweenCompartmentsGlobal = false;				
				}

				if(chemotactByTypeXMlList[j]->findAttribute("SaturationLinearCoef")){
					cd.saturationCoef=chemotactByTypeXMlList[j]->getAttributeAsDouble("SaturationLinearCoef");
					cd.formulaName="SaturationLinearChemotaxisFormula";
				}

				if(chemotactByTypeXMlList[j]->findAttribute("ChemotactTowards")){
					//ASSERT_OR_THROW("ChemotactTowards is deprecated now. Please replace it with ChemotactAtInterfaceWith.",chemotaxisFieldDataVec.size());
					cd.chemotactTowardsTypesString=chemotactByTypeXMlList[j]->getAttribute("ChemotactTowards");
				}else if (chemotactByTypeXMlList[j]->findAttribute("ChemotactAtInterfaceWith")){// both keywords are OK
					cd.chemotactTowardsTypesString=chemotactByTypeXMlList[j]->getAttribute("ChemotactAtInterfaceWith");
				}
				//cerr<<"cd.typeName="<<cd.typeName<<" cd.lambda="<<endl;
				
				
				//jfg:
				if( chemotactByTypeXMlList[j]->findAttribute("PowerCoef") )
				{
					cd.powerLevel = chemotactByTypeXMlList[j]->getAttributeAsDouble("PowerCoef");
					if( cd.powerLevel != 1 )
					{
						cd.formulaName = "PowerChemotaxisFormula";//powerChemotaxisFormula
					}
				}
			//jfg, end
			}
			
		}

	}
	//Now after parsing XMLtree we initialize things

	ASSERT_OR_THROW("You forgot to define the body of chemotaxis plugin. See manual for details",chemotaxisFieldDataVec.size());

	automaton=potts->getAutomaton();

	unsigned char maxType=0;
	//first will find max type value

//	cerr<<"chemotaxisFieldDataVec[0].vecChemotaxisData.size()="<<chemotaxisFieldDataVec[0].vecChemotaxisData.size()<<endl;

	for(int i = 0 ; i < chemotaxisFieldDataVec.size() ; ++ i)
		for(int j = 0 ; j < chemotaxisFieldDataVec[i].vecChemotaxisData.size() ; ++j){
			if( automaton->getTypeId(chemotaxisFieldDataVec[i].vecChemotaxisData[j].typeName) > maxType )
				maxType = automaton->getTypeId(chemotaxisFieldDataVec[i].vecChemotaxisData[j].typeName);
		}

		//make copy vector vecVecChemotaxisData 
		//    std::vector<std::vector<ChemotaxisData> > vecVecChemotaxisDataTmp=vecVecChemotaxisData;
		vecVecChemotaxisData.clear();

		cerr<<"maxType="<<(int)maxType<<endl;
		//now will allocate vectors based on maxType - this will result in t=0 lookup time later...
		//    cerr<<"vecVecChemotaxisDataTmp.size()="<<vecVecChemotaxisDataTmp.size()<<endl;
		//    cerr<<"(int)maxType+1="<<(int)maxType+1<<endl;
		vecVecChemotaxisData.assign(chemotaxisFieldDataVec.size() , vector<ChemotaxisData>((int)maxType+1,ChemotaxisData() ) );


		/*   //now will allocate vectors based on maxType - this will result in t=0 lookup time later...
		flexChemotaxisDataVec.assign(flexChemotaxisDataVecTmp.size(),vector<float>((int)maxType+1,0.0));*/

		vector<string> vecTypeNamesTmp;


		if(chemotaxisAlgorithm=="merks"){
			algorithmPtr=&ChemotaxisPlugin::merksChemotaxis;
		} else if(chemotaxisAlgorithm=="regular") {
			algorithmPtr=&ChemotaxisPlugin::regularChemotaxis;
		}


		for(int i = 0 ; i < chemotaxisFieldDataVec.size() ; ++ i)
			for(int j = 0 ; j < chemotaxisFieldDataVec[i].vecChemotaxisData.size() ; ++j){

				vecTypeNamesTmp.clear();

				int cellTypeId = automaton->getTypeId(chemotaxisFieldDataVec[i].vecChemotaxisData[j].typeName);

				vecVecChemotaxisData[i][cellTypeId]=chemotaxisFieldDataVec[i].vecChemotaxisData[j];

				ChemotaxisData &chemotaxisDataTmp=vecVecChemotaxisData[i][cellTypeId];
				//Mapping type names to type ids
				if(chemotaxisDataTmp.chemotactTowardsTypesString!=""){ //non-empty string we need to parse and process it 

					parseStringIntoList(chemotaxisDataTmp.chemotactTowardsTypesString,vecTypeNamesTmp,",");
					chemotaxisDataTmp.chemotactTowardsTypesString=""; //empty original string it is no longer needed

					for(int idx=0 ; idx < vecTypeNamesTmp.size() ; ++idx){

						unsigned char typeTmp=automaton->getTypeId(vecTypeNamesTmp[idx]);
						chemotaxisDataTmp.chemotactTowardsTypesVec.push_back(typeTmp);

					}
				}


				if(vecVecChemotaxisData[i][cellTypeId].formulaName=="SaturationChemotaxisFormula"){
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::saturationChemotaxisFormula;
				}
				else if( vecVecChemotaxisData[i][cellTypeId].formulaName=="SaturationLinearChemotaxisFormula"){
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::saturationLinearChemotaxisFormula;

				}
				//jfg, more formulas
				
				
				else if ( vecVecChemotaxisData[i][cellTypeId].formulaName == "PowerChemotaxisFormula" )
				{
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::powerChemotaxisFormula;
				} 
				else if ( vecVecChemotaxisData[i][cellTypeId].formulaName == "SaturationDifferenceChemotaxisFormula" )
				{
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::saturationDifferenceChemotaxisFormula;
				}
				else if ( vecVecChemotaxisData[i][cellTypeId].formulaName == "Log10DivisionFormula" )
				{
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::log10DivisionFormula;
				}
				else if ( vecVecChemotaxisData[i][cellTypeId].formulaName == "LogNatDivisionFormula" )
				{
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::logNatDivisionFormula;
				}
				else if ( vecVecChemotaxisData[i][cellTypeId].formulaName == "Log10DifferenceFormula" )
				{
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::log10DifferenceFormula;
				}
				else if ( vecVecChemotaxisData[i][cellTypeId].formulaName == "LogNatDifferenceFormula" )
				{
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::logNatDifferenceFormula;
				}
				
				// jfg, end
				else{
					vecVecChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::simpleChemotaxisFormula;
				}

				cerr<<"i="<<i<<" cellTypeId="<<cellTypeId<<endl;
				vecVecChemotaxisData[i][cellTypeId].outScr();

			}

			//Now need to initialize field pointers
			cerr<<"chemicalFieldSourceVec.size()="<<chemotaxisFieldDataVec.size()<<endl;
			fieldVec.clear();
			fieldVec.assign(chemotaxisFieldDataVec.size(),0);//allocate fieldVec

			fieldNameVec.clear();
		    fieldNameVec.assign(chemotaxisFieldDataVec.size(),"");//allocate fieldNameVec

			for(unsigned int i=0; i < chemotaxisFieldDataVec.size() ; ++i ){
				//if(chemotaxisFieldDataVec[i].chemicalFieldSource=="DiffusionSolverFE")
				{
					map<string,Field3D<float>*> & nameFieldMap = sim->getConcentrationFieldNameMap();
					map<string,Field3D<float>*>::iterator mitr=nameFieldMap.find(chemotaxisFieldDataVec[i].chemicalFieldName);

					if(mitr!=nameFieldMap.end()){
						fieldVec[i]=mitr->second;
						fieldNameVec[i]=chemotaxisFieldDataVec[i].chemicalFieldName;
					}else{
						ASSERT_OR_THROW("No chemical field has been loaded!", fieldVec[i]);

					}
				}
	
			}

}


float ChemotaxisPlugin::simpleChemotaxisFormula(float _flipNeighborConc,float _conc,ChemotaxisData & _chemotaxisData){
	return (_flipNeighborConc-_conc)*_chemotaxisData.lambda;
}

float ChemotaxisPlugin::saturationChemotaxisFormula(float _flipNeighborConc,float _conc,ChemotaxisData & _chemotaxisData){

	return   _chemotaxisData.lambda*(
		_flipNeighborConc/(_chemotaxisData.saturationCoef+_flipNeighborConc)
		-_conc/(_chemotaxisData.saturationCoef+_conc)
		);

}

float ChemotaxisPlugin::saturationLinearChemotaxisFormula(float _flipNeighborConc,float _conc,ChemotaxisData & _chemotaxisData){
	return   _chemotaxisData.lambda*(
		_flipNeighborConc/(_chemotaxisData.saturationCoef*_flipNeighborConc+1)
		-_conc/(_chemotaxisData.saturationCoef*_conc+1)
		);

}

//jfg, more formulas
float ChemotaxisPlugin::saturationDifferenceChemotaxisFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData)
{
	return _chemotaxisData.lambda*(
		( _flipNeighborConc - _conc )/( _chemotaxisData.saturationCoef + _flipNeighborConc + _conc )
		);
	)
	
}

float ChemotaxisPlugin::powerChemotaxisFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData)
{
	float diff = _flipNeighborConc-_conc
	if (_chemotaxisData.powerLevel < 0 && diff == 0)
	{
		return 9E99 * _chemotaxisData.lambda;
	}
	return _chemotaxisData.lambda*pow(
		diff, _chemotaxisData.powerLevel
	);
}

float ChemotaxisPlugin::log10DivisionFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData)
{
	return _chemotaxisData.lambda * log10(
		( 1 + _flipNeighborConc )/( 1 + _conc )
	) ;
}

float ChemotaxisPlugin::logNatDivisionFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData)
{
	return _chemotaxisData.lambda * log(
		( 1 + _flipNeighborConc )/( 1 + _conc )
	) ;
}

float ChemotaxisPlugin::log10DifferenceFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData)
{
	float diff = _flipNeighborConc - _conc
	
	if ( diff <= 0 )
	{
		return -9E99 * _chemotaxisData.lambda;
	}
	
	return _chemotaxisData.lambda * log10( diff );
}

float ChemotaxisPlugin::logNatDifferenceFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData)
{
	float diff = _flipNeighborConc - _conc
	
	if ( diff <= 0 )
	{
		return -9E99 * _chemotaxisData.lambda;
	}
	
	return _chemotaxisData.lambda * log( diff );
}
//jfg, end

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double ChemotaxisPlugin::regularChemotaxis(const Point3D &pt, const CellG *newCell,const CellG *oldCell){

	//    cerr<<"This is regular chemotaxis"<<endl;

	///cells move up the concentration gradient


	/// new cell has to be different than medium i.e. only non-medium cells can chemotact
	///e.g. in chemotaxis only non-medium cell can move a pixel that either belonged to other cell or to medium
	///but situation where medium moves to a new pixel is not considered a chemotaxis

	float energy=0.0;
	std::map<std::string,ChemotaxisData>::iterator mitr;

	if(newCell){
		//       cerr<<"INSIDE NEW CELL CONDITION fieldVec.size()="<<fieldVec.size()<<endl;
		for(unsigned int i = 0 ; i < fieldVec.size() ; ++i){
			bool chemotaxisDone=false;

			//first try "locally defined" chemotaxis

			std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(newCell->extraAttribPtr);
			mitr=chemotaxisDataDictRef.find(fieldNameVec[i]);
			//cerr<<"Looking for field="<<fieldNameVec[i]<<endl;
			ChemotaxisData * chemotaxisDataPtr=0;
			if (mitr!= chemotaxisDataDictRef.end()){
				chemotaxisDataPtr=&mitr->second;

				//newCellFieldNamesVisited.insert(fieldNameVec[i]);
			}
			//cerr<<"chemotaxisDataPtr="<<chemotaxisDataPtr<<endl;
			
			//if(chemotaxisDataPtr )
			//	cerr<<"chemotaxisDataPtr->okToChemotact(oldCell)="<<chemotaxisDataPtr->okToChemotact(oldCell)<<endl;

			if( chemotaxisDataPtr && chemotaxisDataPtr->okToChemotact(oldCell,newCell) ){ 
				// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
				//          cerr<<"BASED ON NEW pt "<<pt<<" oldCell="<<oldCell<<" newCell="<<newCell<<endl;
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
				formulaCurrentPtr=chemotaxisDataPtr->formulaPtr;
				if(formulaCurrentPtr){
					energy+=(this->*formulaCurrentPtr)(fieldVec[i]->get(potts->getFlipNeighbor()) , fieldVec[i]->get(pt) 
						, *chemotaxisDataPtr);
					chemotaxisDone=true;
					
				}
			}


			if( !chemotaxisDone && (int)newCell->type < vecVecChemotaxisData[i].size() ){

				ChemotaxisData & chemotaxisDataRef = vecVecChemotaxisData[i][(int)newCell->type];
				ChemotaxisData * chemotaxisDataPtr = & vecVecChemotaxisData[i][(int)newCell->type];

				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;

				formulaCurrentPtr=chemotaxisDataRef.formulaPtr;


				if( !chemotaxisDataRef.okToChemotact(oldCell,newCell) ){ // chemotaxis id not allowed towards this type of oldCell
					continue;
				}

				if(chemotaxisDataRef.lambda!=0.0){ //THIS IS THE CONDITION THAT TRIGGERS CHEMOTAXIS
					//                   if((int)newCell->type==2){
					//                   cerr<<"pointer to formula="<<formulaCurrentPtr<<endl;
					//                   chemotaxisDataRef.outScr();
					// 
					//                   
					//                   cerr<<"concentration N="<<fieldVec[i]->get(potts->getFlipNeighbor())<<" conc="<<fieldVec[i]->get(pt)<<endl;
					//                   cerr<<"energy="<<(this->*formulaCurrentPtr)(fieldVec[i]->get(potts->getFlipNeighbor()) , fieldVec[i]->get(pt) , chemotaxisDataRef)<<endl;
					// //                   cerr<<"energy="<<simpleChemotaxisFormula(fieldVec[i]->get(potts->getFlipNeighbor()) , fieldVec[i]->get(pt) , chemotaxisDataRef)<<endl;
					//                   }
					//                  energy+=(this->*formulaPtr)(fieldVec[i]->get(potts->getFlipNeighbor()) , fieldVec[i]->get(pt) , chemotaxisDataRef);
					if(formulaCurrentPtr)
						energy+=(this->*formulaCurrentPtr)(fieldVec[i]->get(potts->getFlipNeighbor()) , fieldVec[i]->get(pt) , chemotaxisDataRef);
					


				}


			}

		}
		//       cerr<<"CHEMOTAXIS EN="<<energy<<endl;   
		return energy;
	}
	//   cerr<<"CHEMOTAXIS EN="<<energy<<endl;
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double ChemotaxisPlugin::merksChemotaxis(const Point3D &pt, const CellG *newCell,const CellG *oldCell){

	//    cerr<<"this is merks chemotaxis"<<endl;   
	float energy=0.0;
	std::map<std::string,ChemotaxisData>::iterator mitr;
	//set<string> newCellFieldNamesVisited;
	//set<string> oldCellFieldNamesVisited;


	//    cerr<<"fieldVec.size()="<<fieldVec.size()<<endl;
	for(unsigned int i = 0 ; i < fieldVec.size() ; ++i){
		bool chemotaxisDone=false;
		//first will see if newCell is chemotacting (using locally defined chemotaxis parameters) and if it chemotacts towards oldCell. If yes, then next if statements
		// will be skipped 

		if(newCell){// check if newCell is potentially chemotaxing based on local parameters
			std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(newCell->extraAttribPtr);
			mitr=chemotaxisDataDictRef.find(fieldNameVec[i]);
			//cerr<<"Looking for field="<<fieldNameVec[i]<<endl;
			ChemotaxisData * chemotaxisDataPtr=0;
			if (mitr!= chemotaxisDataDictRef.end()){
				chemotaxisDataPtr=&mitr->second;

				//newCellFieldNamesVisited.insert(fieldNameVec[i]);
			}
			//cerr<<"chemotaxisDataPtr="<<chemotaxisDataPtr<<endl;
			
			//if(chemotaxisDataPtr )
			//	cerr<<"chemotaxisDataPtr->okToChemotact(oldCell)="<<chemotaxisDataPtr->okToChemotact(oldCell)<<endl;

			if( chemotaxisDataPtr && chemotaxisDataPtr->okToChemotact(oldCell,newCell) ){ 
				// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
				//          cerr<<"BASED ON NEW pt "<<pt<<" oldCell="<<oldCell<<" newCell="<<newCell<<endl;
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
				formulaCurrentPtr=chemotaxisDataPtr->formulaPtr;
				if(formulaCurrentPtr){
					energy+=(this->*formulaCurrentPtr)(fieldVec[i]->get(potts->getFlipNeighbor()) , fieldVec[i]->get(pt) 
						, *chemotaxisDataPtr);
				
					chemotaxisDone=true;
					//cerr<<"Energy="<<energy<< " lambda="<<chemotaxisDataPtr->lambda<<endl;
				}
			}
		}
		//first will see if newCell is chemotacting and if it chemotacts towards oldCell. If yes, then next if statement
		// will be skipped and 
		if(!chemotaxisDone && newCell && (int)newCell->type < vecVecChemotaxisData[i].size()){// check if newCell is potentially chemotaxing

			ChemotaxisData & chemotaxisDataRef = vecVecChemotaxisData[i][(int)newCell->type];



			if( chemotaxisDataRef.okToChemotact(oldCell,newCell) && chemotaxisDataRef.lambda!=0.0){ 
				// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
				//          cerr<<"BASED ON NEW pt "<<pt<<" oldCell="<<oldCell<<" newCell="<<newCell<<endl;
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
				formulaCurrentPtr=chemotaxisDataRef.formulaPtr;
				if(formulaCurrentPtr){

			
					energy+=(this->*formulaCurrentPtr)(fieldVec[i]->get(potts->getFlipNeighbor()) , fieldVec[i]->get(pt) 
						, chemotaxisDataRef);
			
					chemotaxisDone=true;
				}
			}
		}

		//now check if old cell chemotaxes locally
		if(!chemotaxisDone && oldCell){
			
			// check if newCell is potentially chemotaxing based on local parameters
			std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(oldCell->extraAttribPtr);
			mitr=chemotaxisDataDictRef.find(fieldNameVec[i]);
			ChemotaxisData * chemotaxisDataPtr=0;
			if (mitr!= chemotaxisDataDictRef.end()){
				chemotaxisDataPtr=&mitr->second;
				//oldCellFieldNamesVisited.insert(fieldNameVec[i]);
			}


			if( chemotaxisDataPtr && chemotaxisDataPtr->okToChemotact(newCell,oldCell) ){ 
				// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
				//          cerr<<"BASED ON NEW pt "<<pt<<" oldCell="<<oldCell<<" newCell="<<newCell<<endl;
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
				formulaCurrentPtr=chemotaxisDataPtr->formulaPtr;
				if(formulaCurrentPtr){


					energy+=(this->*formulaCurrentPtr)(fieldVec[i]->get(potts->getFlipNeighbor()) , fieldVec[i]->get(pt) 
						, *chemotaxisDataPtr);
					chemotaxisDone=true;
			
				}
			}
			

		}

		if(!chemotaxisDone && oldCell && (int)oldCell->type < vecVecChemotaxisData[i].size()){
			//since chemotaxis "based on" newCell did not work we try to see it "based on" oldCell will work
			ChemotaxisData & chemotaxisDataRef = vecVecChemotaxisData[i][(int)oldCell->type];

			if( chemotaxisDataRef.okToChemotact(newCell,oldCell) && chemotaxisDataRef.lambda!=0.0){ 
				// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
				//             cerr<<"BASED ON OLD pt="<<pt<<" oldCell="<<oldCell<<" newCell="<<newCell<<endl;
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
				formulaCurrentPtr=chemotaxisDataRef.formulaPtr;
				if(formulaCurrentPtr){
					energy+=(this->*formulaCurrentPtr)(fieldVec[i]->get(potts->getFlipNeighbor()), fieldVec[i]->get(pt)
						, chemotaxisDataRef);					
					chemotaxisDone=true;
				}
			}
		}

	}

	//    cerr<<"Chemotaxis energy  - Merks alg = "<<energy<<endl;
	return energy;


}

double ChemotaxisPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	//    cerr<<"algorithmPtr="<<algorithmPtr<<endl;



	//    double energy=(this->*algorithmPtr)(pt,newCell,oldCell);
	//    exit(0);
	//    return 0.0;

	//double energy=(this->*algorithmPtr)(pt,newCell,oldCell);
	// cerr<<"Chemotaxis Energy="<<energy<<endl;
	// //return energy;

	return (this->*algorithmPtr)(pt,newCell,oldCell);

}

ChemotaxisData * ChemotaxisPlugin::addChemotaxisData(CellG * _cell,std::string _fieldName){

	ChemotaxisData * chemotaxisDataPtr=getChemotaxisData(_cell,_fieldName);
	if (chemotaxisDataPtr)
		return chemotaxisDataPtr;

	std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(_cell->extraAttribPtr);

	chemotaxisDataDictRef[_fieldName]=ChemotaxisData();
	
	ChemotaxisData & chemotaxisDataRef=chemotaxisDataDictRef[_fieldName];
	chemotaxisDataRef.chemotaxisFormulaDictPtr=&chemotaxisFormulaDict;
	chemotaxisDataRef.automaton=automaton;

	chemotaxisDataRef.formulaPtr=chemotaxisFormulaDict[chemotaxisDataRef.formulaName]; //use simple formula as a default setting
	
	return & chemotaxisDataDictRef[_fieldName];

}

std::vector<std::string> ChemotaxisPlugin::getFieldNamesWithChemotaxisData(CellG * _cell){

	std::vector<std::string> fieldNamesVec;
	std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(_cell->extraAttribPtr);
	for (std::map<std::string,ChemotaxisData>::iterator mitr = chemotaxisDataDictRef.begin() ; mitr!=chemotaxisDataDictRef.end();++mitr){
		fieldNamesVec.push_back(mitr->first);
	}
	return fieldNamesVec;
}

ChemotaxisData * ChemotaxisPlugin::getChemotaxisData(CellG * _cell , std::string _fieldName){
	std::map<std::string,ChemotaxisData>::iterator mitr;
	std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(_cell->extraAttribPtr);
	mitr=chemotaxisDataDictRef.find(_fieldName);
	if (mitr != chemotaxisDataDictRef .end()){
		return &mitr->second;
	}else{
		return 0;
	}
}

std::string ChemotaxisPlugin::toString(){
	return "Chemotaxis";
}


std::string ChemotaxisPlugin::steerableName(){
	return toString();
}

