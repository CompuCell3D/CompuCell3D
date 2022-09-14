#include <CompuCell3D/CC3D.h>

#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>
#include <iostream>
#include <fstream>


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
    chemotaxisFormulaDict["SaturationDifferenceChemotaxisFormula"]=&ChemotaxisPlugin::saturationDifferenceChemotaxisFormula;
    chemotaxisFormulaDict["PowerChemotaxisFormula"]=&ChemotaxisPlugin::powerChemotaxisFormula;
    chemotaxisFormulaDict["Log10DivisionFormula"]=&ChemotaxisPlugin::log10DivisionFormula;
    chemotaxisFormulaDict["LogNatDivisionFormula"]=&ChemotaxisPlugin::logNatDivisionFormula;
    chemotaxisFormulaDict["Log10DifferenceFormula"]=&ChemotaxisPlugin::log10DifferenceFormula;
    chemotaxisFormulaDict["LogNatDifferenceFormula"]=&ChemotaxisPlugin::logNatDifferenceFormula;
    chemotaxisFormulaDict["COMLogScaledChemotaxisFormula"]=&ChemotaxisPlugin::COMLogScaledChemotaxisFormula;
	
}

ChemotaxisPlugin::~ChemotaxisPlugin() {
}


void ChemotaxisPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	xmlData=_xmlData;

	sim = simulator;
	potts = simulator->getPotts();

	// load CenterOfMass plugin if it is not already loaded
	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin = Simulator::pluginManager.get("CenterOfMass", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) plugin->init(simulator);
  
  ExtraMembersGroupAccessorBase * chemotaxisDataAccessorPtr=&chemotaxisDataAccessor;
  ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
  ///************************************************************************************************  
  potts->getCellFactoryGroupPtr()->registerClass(chemotaxisDataAccessorPtr);

	potts->registerEnergyFunctionWithName(this,"Chemotaxis");
	simulator->registerSteerableObject(this);          

}

void ChemotaxisPlugin::extraInit(Simulator *simulator) {

	update(xmlData,true);



}


void ChemotaxisPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	std::vector<ChemotaxisFieldData> chemotaxisFieldDataVec; 


	if(_xmlData->findElement("Algorithm")){
		chemotaxisAlgorithm=_xmlData->getFirstElement("Algorithm")->getText();
		changeToLower(chemotaxisAlgorithm);
	}

	//Parsing ChemicalField Sections
	CC3DXMLElementList chemicalFieldXMlList=_xmlData->getElements("ChemicalField");


	for (int i  = 0 ; i < chemicalFieldXMlList.size() ; ++i){

		chemotaxisFieldDataVec.push_back(ChemotaxisFieldData());
		ChemotaxisFieldData & cfd=chemotaxisFieldDataVec[chemotaxisFieldDataVec.size()-1];

		cfd.chemicalFieldName =chemicalFieldXMlList[i]->getAttribute("Name");

		cfd.vecChemotaxisData.clear();
		//Parsing Chemotaxis by type elements
		CC3DXMLElementList chemotactByTypeXMlList=chemicalFieldXMlList[i]->getElements("ChemotaxisByType");
		CC3D_Log(LOG_TRACE) << "chemotactByTypeXMlList.size()="<<chemotactByTypeXMlList.size();

		for (int j = 0 ; j < chemotactByTypeXMlList.size() ; ++j){
			cfd.vecChemotaxisData.push_back(ChemotaxisData());
			ChemotaxisData & cd=cfd.vecChemotaxisData[cfd.vecChemotaxisData.size()-1];
			cd.typeName=chemotactByTypeXMlList[j]->getAttribute("Type");
			
			//jfg, now that there are a bunch of formulas it'd be good to have ifs to select them,
            // instead of relying on
			// variable names that migth or might not be in the xml
			
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
				else if (cd.formulaName == "SaturationLinearChemotaxisFormula" || 
					cd.formulaName == "SaturationChemotaxisFormula" ||
					cd.formulaName == "SaturationDifferenceChemotaxisFormula" || 
					cd.formulaName == "COMLogScaledChemotaxisFormula")
				{
					CC3D_Log(LOG_WARNING) << "You've asked for a saturation formula but did not provide a saturation coefficient" ;
					exit(0);
				}
				
				if (chemotactByTypeXMlList[j]->findAttribute("DisallowChemotaxisBetweenCompartments")) 
				{
					cd.allowChemotaxisBetweenCompartmentsGlobal = false;	
				}
				
				if(chemotactByTypeXMlList[j]->findAttribute("ChemotactTowards")){
					cd.chemotactTowardsTypesString=chemotactByTypeXMlList[j]->getAttribute("ChemotactTowards");
				}else if (chemotactByTypeXMlList[j]->findAttribute("ChemotactAtInterfaceWith")){// both keywords are OK
					cd.chemotactTowardsTypesString=chemotactByTypeXMlList[j]->getAttribute("ChemotactAtInterfaceWith");
				}
				
				if( chemotactByTypeXMlList[j]->findAttribute("PowerCoef") )
				{
					cd.powerLevel = chemotactByTypeXMlList[j]->getAttributeAsDouble("PowerCoef");
					if( cd.powerLevel == 1.0 )
					{
						cd.formulaName="none";
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
					cd.formulaName="SaturationChemotaxisFormula";//
				}

				if (chemotactByTypeXMlList[j]->findAttribute("DisallowChemotaxisBetweenCompartments")) {
					cd.allowChemotaxisBetweenCompartmentsGlobal = false;				
				}

				if(chemotactByTypeXMlList[j]->findAttribute("SaturationLinearCoef")){
					cd.saturationCoef=chemotactByTypeXMlList[j]->getAttributeAsDouble("SaturationLinearCoef");
					cd.formulaName="SaturationLinearChemotaxisFormula";
				}

				if(chemotactByTypeXMlList[j]->findAttribute("ChemotactTowards")){
					cd.chemotactTowardsTypesString=chemotactByTypeXMlList[j]->getAttribute("ChemotactTowards");
				}else if (chemotactByTypeXMlList[j]->findAttribute("ChemotactAtInterfaceWith")){// both keywords are OK
					cd.chemotactTowardsTypesString=chemotactByTypeXMlList[j]->getAttribute("ChemotactAtInterfaceWith");
				}

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

				if (chemotactByTypeXMlList[j]->findAttribute("LogScaledCoef")){
					cd.saturationCoef = (float)chemotactByTypeXMlList[j]->getAttributeAsDouble("LogScaledCoef");
					cd.formulaName = "COMLogScaledChemotaxisFormula";
				}
			}
			
		}

	}
	//Now after parsing XMLtree we initialize things

	if (!chemotaxisFieldDataVec.size())
		throw CC3DException("You forgot to define the body of chemotaxis plugin. See manual for details");

	automaton=potts->getAutomaton();

	unsigned char maxType=0;
	//first will find max type value

	for(int i = 0 ; i < chemotaxisFieldDataVec.size() ; ++ i)
		for(int j = 0 ; j < chemotaxisFieldDataVec[i].vecChemotaxisData.size() ; ++j){
			if( automaton->getTypeId(chemotaxisFieldDataVec[i].vecChemotaxisData[j].typeName) > maxType )
				maxType = automaton->getTypeId(chemotaxisFieldDataVec[i].vecChemotaxisData[j].typeName);
		}

		vecMapChemotaxisData.clear();

		CC3D_Log(LOG_DEBUG) << "maxType=" << (int)maxType;

		vecMapChemotaxisData.assign(chemotaxisFieldDataVec.size() , unordered_map<unsigned char, ChemotaxisData>() );

		vector<string> vecTypeNamesTmp;


		if(chemotaxisAlgorithm=="merks"){
			algorithmPtr=&ChemotaxisPlugin::merksChemotaxis;
		} else if(chemotaxisAlgorithm=="regular") {
			algorithmPtr=&ChemotaxisPlugin::regularChemotaxis;
		} else if (chemotaxisAlgorithm=="reciprocated") {
			algorithmPtr=&ChemotaxisPlugin::reciprocatedChemotaxis;
		}


		for(int i = 0 ; i < chemotaxisFieldDataVec.size() ; ++ i)
			for(int j = 0 ; j < chemotaxisFieldDataVec[i].vecChemotaxisData.size() ; ++j){

				vecTypeNamesTmp.clear();

				unsigned char cellTypeId = automaton->getTypeId(chemotaxisFieldDataVec[i].vecChemotaxisData[j].typeName);

				vecMapChemotaxisData[i][cellTypeId]=chemotaxisFieldDataVec[i].vecChemotaxisData[j];

				ChemotaxisData &chemotaxisDataTmp=vecMapChemotaxisData[i][cellTypeId];
				//Mapping type names to type ids
				if(chemotaxisDataTmp.chemotactTowardsTypesString!=""){ //non-empty string we need to parse and process it 

					parseStringIntoList(chemotaxisDataTmp.chemotactTowardsTypesString,vecTypeNamesTmp,",");
					chemotaxisDataTmp.chemotactTowardsTypesString=""; //empty original string it is no longer needed

					for(int idx=0 ; idx < vecTypeNamesTmp.size() ; ++idx){

						unsigned char typeTmp=automaton->getTypeId(vecTypeNamesTmp[idx]);
						chemotaxisDataTmp.chemotactTowardsTypesVec.push_back(typeTmp);

					}
				}


				if(vecMapChemotaxisData[i][cellTypeId].formulaName=="SaturationChemotaxisFormula"){
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::saturationChemotaxisFormula;
				}
				else if( vecMapChemotaxisData[i][cellTypeId].formulaName=="SaturationLinearChemotaxisFormula"){
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::saturationLinearChemotaxisFormula;

				}
				//jfg, more formulas
				
				
				else if ( vecMapChemotaxisData[i][cellTypeId].formulaName == "PowerChemotaxisFormula" )
				{
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::powerChemotaxisFormula;
				} 
				else if ( vecMapChemotaxisData[i][cellTypeId].formulaName == "SaturationDifferenceChemotaxisFormula" )
				{
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::saturationDifferenceChemotaxisFormula;
				}
				else if ( vecMapChemotaxisData[i][cellTypeId].formulaName == "Log10DivisionFormula" )
				{
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::log10DivisionFormula;
				}
				else if ( vecMapChemotaxisData[i][cellTypeId].formulaName == "LogNatDivisionFormula" )
				{
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::logNatDivisionFormula;
				}
				else if ( vecMapChemotaxisData[i][cellTypeId].formulaName == "Log10DifferenceFormula" )
				{
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::log10DifferenceFormula;
				}
				else if ( vecMapChemotaxisData[i][cellTypeId].formulaName == "LogNatDifferenceFormula" )
				{
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::logNatDifferenceFormula;
				}
				else if (vecMapChemotaxisData[i][cellTypeId].formulaName == "COMLogScaledChemotaxisFormula")
				{
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::COMLogScaledChemotaxisFormula;
				}
				
				// jfg, end
				else{
					vecMapChemotaxisData[i][cellTypeId].formulaPtr=&ChemotaxisPlugin::simpleChemotaxisFormula;
				}
				CC3D_Log(LOG_DEBUG) << "i="<<i<<" cellTypeId="<<cellTypeId;
				vecMapChemotaxisData[i][cellTypeId].outScr();

			}

			//Now need to initialize field pointers
			CC3D_Log(LOG_DEBUG) << "chemicalFieldSourceVec.size()="<<chemotaxisFieldDataVec.size();
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
					}else if (!fieldVec[i]){
						throw CC3DException("No chemical field has been loaded!");

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

	return _chemotaxisData.lambda*( _flipNeighborConc - _conc )/( _chemotaxisData.saturationCoef + _flipNeighborConc - _conc );
	
	
}

float ChemotaxisPlugin::powerChemotaxisFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData)
{
	float diff = _flipNeighborConc - _conc;


	if (_chemotaxisData.powerLevel < 0 && diff == 0)//don't want NANs or infs going around
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
	float diff = _flipNeighborConc - _conc;


	if ( diff <= 0 )//don't wan't NANs or infs going around
	{
		return -9E99 * _chemotaxisData.lambda;
	}

	return _chemotaxisData.lambda * log10( diff );
}

float ChemotaxisPlugin::logNatDifferenceFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData)
{
	float diff = _flipNeighborConc - _conc;

	if ( diff <= 0 )//don't want NANs or infs going around
	{
		return -9E99 * _chemotaxisData.lambda;
	}

	return _chemotaxisData.lambda * log( diff );
}

float ChemotaxisPlugin::COMLogScaledChemotaxisFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData) 
{
	float den = _chemotaxisData.saturationCoef + _chemotaxisData.concCOM;
	if (den == 0.0) den = std::numeric_limits<float>::epsilon();
	return (_flipNeighborConc-_conc) * _chemotaxisData.lambda / den;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double ChemotaxisPlugin::regularChemotaxis(const Point3D &pt, const CellG *newCell,const CellG *oldCell){


	///cells move up the concentration gradient


	/// new cell has to be different than medium i.e. only non-medium cells can chemotact
	///e.g. in chemotaxis only non-medium cell can move a pixel that either belonged to other cell or to medium
	///but situation where medium moves to a new pixel is not considered a chemotaxis

	float energy=0.0;
	std::map<std::string,ChemotaxisData>::iterator mitr;

	if(newCell){

		for(unsigned int i = 0 ; i < fieldVec.size() ; ++i){
			bool chemotaxisDone=false;
			auto field = fieldVec[i];

			//first try "locally defined" chemotaxis

			std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(newCell->extraAttribPtr);
			mitr=chemotaxisDataDictRef.find(fieldNameVec[i]);

			ChemotaxisData * chemotaxisDataPtr=0;
			if (mitr!= chemotaxisDataDictRef.end()){
				chemotaxisDataPtr=&mitr->second;

			}

			if( chemotaxisDataPtr && chemotaxisDataPtr->okToChemotact(oldCell,newCell) ){ 
				// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
				formulaCurrentPtr=chemotaxisDataPtr->formulaPtr;

				if(formulaCurrentPtr){
					if (formulaCurrentPtr == &ChemotaxisPlugin::COMLogScaledChemotaxisFormula)
						chemotaxisDataPtr->concCOM = field->get(Point3D(newCell->xCOM, newCell->yCOM, newCell->zCOM));
					
					energy+=(this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()) , field->get(pt), *chemotaxisDataPtr);
					chemotaxisDone=true;
					
				}
			}


			if( !chemotaxisDone ){

				auto itr = vecMapChemotaxisData[i].find(newCell->type);

				if (itr == vecMapChemotaxisData[i].end()) continue;

				ChemotaxisData & chemotaxisDataRef = itr->second;
				ChemotaxisData * chemotaxisDataPtr = &itr->second;

				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;

				formulaCurrentPtr=chemotaxisDataRef.formulaPtr;

				// chemotaxis id not allowed towards this type of oldCell
				if( !chemotaxisDataRef.okToChemotact(oldCell,newCell) ){
					continue;
				}

				//THIS IS THE CONDITION THAT TRIGGERS CHEMOTAXIS
				if(chemotaxisDataRef.lambda!=0.0){
					if(formulaCurrentPtr) {
						if (formulaCurrentPtr == &ChemotaxisPlugin::COMLogScaledChemotaxisFormula)
							chemotaxisDataRef.concCOM = field->get(Point3D(newCell->xCOM, newCell->yCOM, newCell->zCOM));
						
						energy+=(this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()) , field->get(pt) , chemotaxisDataRef);
					}
					


				}


			}

		}
		return energy;
	}

	return 0;
}

double ChemotaxisPlugin::reciprocatedChemotaxis(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
	// Equivalent to regularChemotaxis(pt, newCell, oldCell) +
    // regularChemotaxis(pt, oldCell, newCell), but in a single loop over fields
	// return regularChemotaxis(pt, newCell, oldCell) + regularChemotaxis(pt, oldCell, newCell);

	double energy = 0;

	std::map<std::string,ChemotaxisData>::iterator mitr;

	for(unsigned int i = 0; i < fieldVec.size(); ++i){
		auto field = fieldVec[i];

		if(newCell){
			bool chemotaxisDone = false;

			// first try "locally defined" chemotaxis
			std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(newCell->extraAttribPtr);
			mitr = chemotaxisDataDictRef.find(fieldNameVec[i]);
			
			ChemotaxisData * chemotaxisDataPtr = 0;
			if (mitr != chemotaxisDataDictRef.end()) chemotaxisDataPtr=&mitr->second;
			
			// when chemotaxis is allowed towards this type of oldCell and lambda is non-zero
			if(chemotaxisDataPtr && chemotaxisDataPtr->okToChemotact(oldCell, newCell)){ 
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr = 0;
				formulaCurrentPtr=chemotaxisDataPtr->formulaPtr;
				if(formulaCurrentPtr){
					energy += (this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()), field->get(pt), *chemotaxisDataPtr);
					chemotaxisDone = true;
				}
			}

			if(!chemotaxisDone ){

				auto itr = vecMapChemotaxisData[i].find(newCell->type);

				if (itr != vecMapChemotaxisData[i].end()) {

					ChemotaxisData & chemotaxisDataRef = itr->second;
					ChemotaxisData * chemotaxisDataPtr = & itr->second;
					ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr = 0;

					formulaCurrentPtr = chemotaxisDataRef.formulaPtr;

					if(chemotaxisDataRef.okToChemotact(oldCell, newCell) && chemotaxisDataRef.lambda!=0.0 && formulaCurrentPtr)
						energy += (this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()), field->get(pt), chemotaxisDataRef);

				}
			}
		}
		if(oldCell){
			bool chemotaxisDone=false;

			// first try "locally defined" chemotaxis
			std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(oldCell->extraAttribPtr);
			mitr = chemotaxisDataDictRef.find(fieldNameVec[i]);
			
			ChemotaxisData * chemotaxisDataPtr = 0;
			if (mitr != chemotaxisDataDictRef.end()) chemotaxisDataPtr=&mitr->second;
			
			// when chemotaxis is allowed towards this type of newCell and lambda is non-zero
			if(chemotaxisDataPtr && chemotaxisDataPtr->okToChemotact(newCell, oldCell)){ 
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr = 0;
				formulaCurrentPtr = chemotaxisDataPtr->formulaPtr;
				if(formulaCurrentPtr){
					energy += (this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()), field->get(pt), *chemotaxisDataPtr);
					chemotaxisDone = true;
				}
			}

			if(!chemotaxisDone){

				auto itr = vecMapChemotaxisData[i].find(oldCell->type);

				if (itr != vecMapChemotaxisData[i].end()) {

					ChemotaxisData & chemotaxisDataRef = itr->second;
					ChemotaxisData * chemotaxisDataPtr = & itr->second;
					ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr = 0;

					formulaCurrentPtr=chemotaxisDataRef.formulaPtr;

					if(chemotaxisDataRef.okToChemotact(newCell, oldCell) && chemotaxisDataRef.lambda!=0.0 && formulaCurrentPtr)
						energy += (this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()), field->get(pt), chemotaxisDataRef);

				}
			}
		}
	}

	return energy;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double ChemotaxisPlugin::merksChemotaxis(const Point3D &pt, const CellG *newCell,const CellG *oldCell){

	float energy=0.0;
	std::map<std::string,ChemotaxisData>::iterator mitr;

	for(unsigned int i = 0 ; i < fieldVec.size() ; ++i){
		bool chemotaxisDone=false;
		auto field = fieldVec[i];
		//first will see if newCell is chemotacting (using locally defined chemotaxis parameters)
        // and if it chemotacts towards oldCell. If yes, then next if statements
		// will be skipped 

		if(newCell){// check if newCell is potentially chemotaxing based on local parameters
			std::map<std::string,ChemotaxisData> & chemotaxisDataDictRef = *chemotaxisDataAccessor.get(newCell->extraAttribPtr);
			mitr=chemotaxisDataDictRef.find(fieldNameVec[i]);

			ChemotaxisData * chemotaxisDataPtr=0;
			if (mitr!= chemotaxisDataDictRef.end()){
				chemotaxisDataPtr=&mitr->second;
			}

			if( chemotaxisDataPtr && chemotaxisDataPtr->okToChemotact(oldCell,newCell) ){ 
				// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
				formulaCurrentPtr=chemotaxisDataPtr->formulaPtr;
				if(formulaCurrentPtr){
					if(formulaCurrentPtr == &ChemotaxisPlugin::COMLogScaledChemotaxisFormula)
						chemotaxisDataPtr->concCOM = field->get(Point3D(newCell->xCOM, newCell->yCOM, newCell->zCOM));
					
					energy+=(this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()) , field->get(pt), *chemotaxisDataPtr);
				
					chemotaxisDone=true;
				}
			}
		}
		//first will see if newCell is chemotacting and if it chemotacts towards oldCell.
        // If yes, then next if statement will be skipped
		if(!chemotaxisDone && newCell){// check if newCell is potentially chemotaxing

			auto itr = vecMapChemotaxisData[i].find(newCell->type);

			if (itr != vecMapChemotaxisData[i].end()) {

				ChemotaxisData & chemotaxisDataRef = itr->second;

				if( chemotaxisDataRef.okToChemotact(oldCell,newCell) && chemotaxisDataRef.lambda!=0.0){
					// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
					ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
					formulaCurrentPtr=chemotaxisDataRef.formulaPtr;
					if(formulaCurrentPtr){
						if(formulaCurrentPtr == &ChemotaxisPlugin::COMLogScaledChemotaxisFormula)
							chemotaxisDataRef.concCOM = field->get(Point3D(newCell->xCOM, newCell->yCOM, newCell->zCOM));

						energy+=(this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()) , field->get(pt), chemotaxisDataRef);

						chemotaxisDone=true;
					}
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
			}


			if( chemotaxisDataPtr && chemotaxisDataPtr->okToChemotact(newCell,oldCell) ){ 
				// chemotaxis is allowed towards this type of oldCell and lambda is non-zero

				ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
				formulaCurrentPtr=chemotaxisDataPtr->formulaPtr;
				if(formulaCurrentPtr){
					if(formulaCurrentPtr == &ChemotaxisPlugin::COMLogScaledChemotaxisFormula)
						chemotaxisDataPtr->concCOM = field->get(Point3D(oldCell->xCOM, oldCell->yCOM, oldCell->zCOM));

					energy+=(this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()) , field->get(pt), *chemotaxisDataPtr);
					chemotaxisDone=true;
			
				}
			}
		}

		if(!chemotaxisDone && oldCell){

			auto itr = vecMapChemotaxisData[i].find(oldCell->type);

			if (itr != vecMapChemotaxisData[i].end()) {

				//since chemotaxis "based on" newCell did not work we try to see it "based on" oldCell will work
				ChemotaxisData & chemotaxisDataRef = itr->second;

				if( chemotaxisDataRef.okToChemotact(newCell,oldCell) && chemotaxisDataRef.lambda!=0.0){
					// chemotaxis is allowed towards this type of oldCell and lambda is non-zero
					CC3D_Log(LOG_TRACE) << "BASED ON OLD pt="<<pt<<" oldCell="<<oldCell<<" newCell="<<newCell;
					ChemotaxisPlugin::chemotaxisEnergyFormulaFcnPtr_t formulaCurrentPtr=0;
					formulaCurrentPtr=chemotaxisDataRef.formulaPtr;
					if(formulaCurrentPtr){
						if(formulaCurrentPtr == &ChemotaxisPlugin::COMLogScaledChemotaxisFormula)
							chemotaxisDataRef.concCOM = field->get(Point3D(oldCell->xCOM, oldCell->yCOM, oldCell->zCOM));

						energy+=(this->*formulaCurrentPtr)(field->get(potts->getFlipNeighbor()), field->get(pt), chemotaxisDataRef);
						chemotaxisDone=true;
					}
				}

			}
		}

	}

	return energy;


}

double ChemotaxisPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

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

