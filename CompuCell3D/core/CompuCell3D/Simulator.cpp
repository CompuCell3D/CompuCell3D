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


#include "ClassRegistry.h"
using namespace CompuCell3D;

#include <CompuCell3D/Boundary/BoundaryStrategy.h>

#include <CompuCell3D/Potts3D/EnergyFunctionCalculator.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <PublicUtilities/StringUtils.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <string>
#include <CompuCell3D/Serializer.h>

#include <time.h>
#include <limits>
#include <sstream>

#include <XMLUtils/CC3DXMLElement.h>

#ifdef QT_WRAPPERS_AVAILABLE
	#include <QtWrappers/StreamRedirectors/CustomStreamBuffers.h>
#endif


#undef max
#undef min


#include "Simulator.h"


#define DEBUG_IN "("<<in.getLineNumber()<<" , "<<in.getColumnNumber()<<")"<<endl

using namespace std;

PluginManager<Plugin> Simulator::pluginManager;
PluginManager<Steppable> Simulator::steppableManager;
BasicPluginManager<PluginBase> Simulator::pluginBaseManager;


Simulator::Simulator() :
cerrStreamBufOrig(0),
coutStreamBufOrig(0),
qStreambufPtr(0),
restartEnabled(false)
{
	newPlayerFlag=false;
	ppdPtr=0;
	ppdCC3DPtr=0;
	readPottsSectionFromXML=false;
	simValue=20.5;
	pluginManager.setSimulator(this);
	steppableManager.setSimulator(this);
	currstep=-1;
	classRegistry = new ClassRegistry(this);
	pUtils=new ParallelUtilsOpenMP();
    pUtilsSingle=new ParallelUtilsOpenMP();

	simulatorIsStepping=false;
	potts.setSimulator(this);
	cerrStreamBufOrig=cerr.rdbuf();
	coutStreamBufOrig=cout.rdbuf();

	//qStreambufPtr=getQTextEditBuffer();

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Simulator::~Simulator() {
	// cerr<<"\n\n\n********************************************************************************"<<endl;
	// cerr<<"\n\n\n\n INSIDE SIMULATOR DELETE \n\n\n\n "<<endl;
	// cerr<<"\n\n\n********************************************************************************"<<endl;

	delete classRegistry;
	delete pUtils;
    delete pUtilsSingle;

	cerr << "Simulator: extra destroy for boundary strategy" << endl;
	BoundaryStrategy::destroy();

#ifdef QT_WRAPPERS_AVAILABLE
	//restoring original cerr stream buffer
	if (cerrStreamBufOrig)
		cerr.rdbuf(cerrStreamBufOrig);

	if (qStreambufPtr)
		delete qStreambufPtr;
#endif

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ptrdiff_t Simulator::getCerrStreamBufOrig(){
	return (ptrdiff_t)(void *)cerrStreamBufOrig;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Simulator::restoreCerrStreamBufOrig(ptrdiff_t _ptr){
	cerr.rdbuf((streambuf *)(void *)_ptr);
	//////cerr.rdbuf((streambuf *)_ptr);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Simulator::setOutputRedirectionTarget(ptrdiff_t  _ptr){

	//CustomStreamBufferFactory bufferFactory;
	//qStreambufPtr=bufferFactory.getQTextEditBuffer();

//we may also try to implement buffer switching during player runtime so that it does not require player restart
//for now it is ok to have this basic type of switching
#ifdef QT_WRAPPERS_AVAILABLE


	//if (!qStreambufPtr)
	//	return ; //means setOutputRedirectionTarget has already been called

	if (_ptr<0){//output goes to system console

		cerr.rdbuf(cerrStreamBufOrig);
		return;
	}

	if (!_ptr){ //do not output anything at all
		cerr.rdbuf(0);
		return;
	}

	qStreambufPtr = new QTextEditBuffer;


	qStreambufPtr->setQTextEditPtr((void*)_ptr);
	cerr.rdbuf(qStreambufPtr); //redirecting output to the external target
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
BoundaryStrategy * Simulator::getBoundaryStrategy(){
	return BoundaryStrategy::getInstance();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Simulator::registerConcentrationField(std::string _name,Field3D<float>* _fieldPtr){
	concentrationFieldNameMap.insert(std::make_pair(_name,_fieldPtr));
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> Simulator::getConcentrationFieldNameVector(){
	vector<string> fieldNameVec;
	std::map<std::string,Field3D<float>*>::iterator mitr;
	for (mitr=concentrationFieldNameMap.begin()  ; mitr !=concentrationFieldNameMap.end() ; ++mitr){
		fieldNameVec.push_back(mitr->first);
	}
	return fieldNameVec;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Field3D<float>* Simulator::getConcentrationFieldByName(std::string _fieldName){
		//this function crashes CC3D when called from outside Simulator. 
      std::map<std::string,Field3D<float>*> & fieldMap=this->getConcentrationFieldNameMap();
	  //cerr<<" mapSize="<<fieldMap.size()<<endl;
      std::map<std::string,Field3D<float>*>::iterator mitr;
      mitr=fieldMap.find(_fieldName);
      if(mitr!=fieldMap.end()){
         return mitr->second;
      }else{
         return 0;
      }


	//cerr<<"LOOKING FOR FIELD "<<_fieldName<<endl;
	//cerr<<" mapSize="<<concentrationFieldNameMap.size()<<endl;
	//std::map<std::string,Field3DImpl<float>*>::iterator mitr=concentrationFieldNameMap.find(_fieldName);

	//if(mitr!=concentrationFieldNameMap.end()){
	//	cerr<<" GOT NON ZERO PTR="<<mitr->second<<endl;
	//	return mitr->second;
	//}
	//else{
	//	cerr<<" GOT ZERO PTR"<<endl;
	//	return 0;
	//}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Simulator::serialize(){
	for(size_t i = 0 ; i < serializerVec.size() ; ++i){
		serializerVec[i]->serialize();
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Simulator::registerSteerableObject(SteerableObject * _steerableObject){
	// cerr<<"Dealing with _steerableObject->steerableName()="<<_steerableObject->steerableName()<<endl;
	std::map<std::string,SteerableObject *>::iterator mitr;
	mitr=steerableObjectMap.find(_steerableObject->steerableName());
	// cerr<<"after find"<<endl;

	ASSERT_OR_THROW("Steerable Object "+_steerableObject->steerableName()+" already exist!",  mitr==steerableObjectMap.end());

	steerableObjectMap[_steerableObject->steerableName()]=_steerableObject;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Simulator::unregisterSteerableObject(const std::string & _objectName){
	std::map<std::string,SteerableObject *>::iterator mitr;
	mitr=steerableObjectMap.find(_objectName);
	if(mitr!=steerableObjectMap.end()){
		steerableObjectMap.erase(mitr);
	}else{
		cerr<<"Could not find steerable object called "<<_objectName<<endl;
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SteerableObject * Simulator::getSteerableObject(const std::string & _objectName){
	std::map<std::string,SteerableObject *>::iterator mitr;
	mitr=steerableObjectMap.find(_objectName);
	if(mitr!=steerableObjectMap.end()){
		return mitr->second;
	}else{
		return 0;
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Simulator::postEvent(CC3DEvent & _ev){
	//cerr<<"INSIDE SIMULATOR::postEvent"<<endl;
		pUtils->handleEvent(_ev); //let parallel utils konw about all events

		string pluginName;
		BasicPluginManager<Plugin>::infos_t *infos = &pluginManager.getPluginInfos();
		BasicPluginManager<Plugin>::infos_t::iterator it;
		//for (it = infos->begin(); it != infos->end(); it++)	{
		//	cerr<<" THIS IS PLUGIN NAME "<<(*it)->getName()<<endl;
		//}

		for (it = infos->begin(); it != infos->end(); it++){
			pluginName=(*it)->getName();
			if (pluginManager.isLoaded((*it)->getName())) {
				Plugin *plugin = pluginManager.get((*it)->getName());
				plugin->handleEvent(_ev);
			}

		}

		string steppableName;
		BasicPluginManager<Steppable>::infos_t *infos_step = &steppableManager.getPluginInfos();
		BasicPluginManager<Steppable>::infos_t::iterator it_step;
		for (it_step = infos_step->begin(); it_step != infos_step->end(); it_step++){

			steppableName=(*it_step)->getName();
			// cerr<<"processign steppable="<<steppableName<<endl;

			if (steppableManager.isLoaded(steppableName)) {
				// cerr<<"SENDING EVENT TO THE STEPPABLE "<<steppableName<<endl;
				Steppable *steppable= steppableManager.get(steppableName);
				steppable->handleEvent(_ev);
			}
		}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Simulator::start() {

	try{
		// Print the names of loaded plugins
		cerr << "Simulator::start():  Plugins:";
		BasicPluginManager<Plugin>::infos_t *infos = &pluginManager.getPluginInfos();
		BasicPluginManager<Plugin>::infos_t::iterator it;
		for (it = infos->begin(); it != infos->end(); it++)
			if (pluginManager.isLoaded((*it)->getName())) {
				if (it != infos->begin()) cerr << ",";
				cerr << " " << (*it)->getName();
			}
			cerr << endl;


			classRegistry->start();

			currstep = 0;
			// Output statisitcs
			cerr << "Step " << 0 << " "
				<< "Energy " << potts.getEnergy() << " "
				<< "Cells " << potts.getNumCells()
				<< endl;

			simulatorIsStepping=true; //initialize flag that simulator is stepping
	}catch (const BasicException &e) {
		cerr << "ERROR: " << e << endl;
		unloadModules();
		stringstream errorMessageStream;

		errorMessageStream<<"Exception in C++ code :\n"<<e.getMessage()<<"\n"<<"Location \n"<<"FILE :"<<e.getLocation().getFilename()<<"\n"<<"LINE :"<<e.getLocation().getLine();
		recentErrorMessage=errorMessageStream.str();
		cerr<<"THIS IS recentErrorMessage="<<recentErrorMessage<<endl;
		if (!newPlayerFlag){
			throw e;
		}
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Simulator::extraInit(){

	try{
		BasicPluginManager<Plugin>::infos_t *infos = &pluginManager.getPluginInfos();
		BasicPluginManager<Plugin>::infos_t::iterator it;

        cerr<<"begin extraInit calls for plugins"<<endl;
		for (it = infos->begin(); it != infos->end(); it++)
			if (pluginManager.isLoaded((*it)->getName())) {
				//pluginManager.get((*it)->getName())->extraInit(this);
				//if (it != infos->begin()) cerr << ",";
//				cerr << " extraInit for: " << (*it)->getName() << endl;
				pluginManager.get((*it)->getName())->extraInit(this);
//				cerr << " DONE extraInit for: " << (*it)->getName() << endl;

			}

        cerr<<"finish extraInit calls for plugins"<<endl;
		classRegistry->extraInit(this);

	}catch (const BasicException &e) {
		cerr << "ERROR: " << e << endl;
		unloadModules();
		stringstream errorMessageStream;

		errorMessageStream<<"Exception in C++ code :\n"<<e.getMessage()<<"\n"<<"Location \n"<<"FILE :"<<e.getLocation().getFilename()<<"\n"<<"LINE :"<<e.getLocation().getLine();
		recentErrorMessage=errorMessageStream.str();
		cerr<<"THIS IS recentErrorMessage="<<recentErrorMessage<<endl;
		if (!newPlayerFlag){
			throw e;
		}


	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Simulator::step(const unsigned int currentStep) {

	try{
		//potts is initialized in readXML - so is most of other Steppables etc.

		//    for (std::map<std::string,SteerableObject *>::iterator mitr=steerableObjectMap.begin() ; mitr!=steerableObjectMap.end() ; ++mitr){
		//       cerr<<"Module "<<mitr->first <<" toString() "<< mitr->second->toString()<<endl;
		//    }
		// Run potts metropolis
		Dim3D dim = potts.getCellFieldG()->getDim();
		int flipAttempts = (int)(dim.x * dim.y * dim.z * ppdCC3DPtr->flip2DimRatio); //may be a member
		int flips = potts.metropolis(flipAttempts, ppdCC3DPtr->temperature);

		currstep = currentStep;

		// Run steppables -after metropilis function is finished you sweep over classRegistry - mainly for outputing purposes
		classRegistry->step(currentStep);

		// Output statisitcs
		if(ppdCC3DPtr->debugOutputFrequency && ! (currentStep % ppdCC3DPtr->debugOutputFrequency) ){
			cerr << "Step " << currentStep << " "
				<< "Flips " << flips << "/" << flipAttempts << " "
				<< "Energy " << potts.getEnergy() << " "
				<< "Cells " << potts.getNumCells()<<" Inventory="<<potts.getCellInventory().getCellInventorySize()
				<< endl;
		}

	}catch (const BasicException &e) {
		cerr << "ERROR: " << e << endl;
		unloadModules();
		stringstream errorMessageStream;

		errorMessageStream<<"Exception in C++ code :\n"<<e.getMessage()<<"\n"<<"Location \n"<<"FILE :"<<e.getLocation().getFilename()<<"\n"<<"LINE :"<<e.getLocation().getLine();
		recentErrorMessage=errorMessageStream.str();
		cerr<<"THIS IS recentErrorMessage="<<recentErrorMessage<<endl;
		if (!newPlayerFlag){
			throw e;
		}

	}

}



void Simulator::finish() {

	try{
		//cerr<<"inside finish"<<endl;
		ppdCC3DPtr->temperature = 0.0;
		//cerr<<"inside finish 1"<<endl;

		for (unsigned int i = 1; i <= ppdCC3DPtr->anneal; i++)
			step(ppdCC3DPtr->numSteps+i);
		//cerr<<"inside finish 2"<<endl;
		classRegistry->finish();
		unloadModules();
		//cerr<<"inside finish 3"<<endl;

	}catch (const BasicException &e) {
		cerr << "ERROR: " << e << endl;
		stringstream errorMessageStream;

		errorMessageStream<<"Exception in C++ code :\n"<<e.getMessage()<<"\n"<<"Location \n"<<"FILE :"<<e.getLocation().getFilename()<<"\n"<<"LINE :"<<e.getLocation().getLine();
		recentErrorMessage=errorMessageStream.str();
		cerr<<"THIS IS recentErrorMessage="<<recentErrorMessage<<endl;
		if (!newPlayerFlag){
			throw e;
		}

	}

}
void Simulator::cleanAfterSimulation(){

	potts.getCellInventory().cleanInventory();
	unloadModules();
}
void Simulator::unloadModules(){
	pluginManager.unload();
	steppableManager.unload();
}

void Simulator::processMetadataCC3D(CC3DXMLElement * _xmlData){
		if (!_xmlData) //no metadata were specified
			return;
		if (_xmlData->getFirstElement("NumberOfProcessors")) {
			unsigned int numberOfProcessors=_xmlData->getFirstElement("NumberOfProcessors")->getUInt();
			pUtils->setNumberOfWorkNodes(numberOfProcessors);
			CC3DEventChangeNumberOfWorkNodes workNodeChangeEvent;
			workNodeChangeEvent.newNumberOfNodes=numberOfProcessors;

			// this will cause redundant calculations inside pUtils but since we do not call it often it is ok . This way code remains cleaner
			postEvent(workNodeChangeEvent);

		}else if(_xmlData->getFirstElement("VirtualProcessingUnits")){

			unsigned int numberOfVPUs=_xmlData->getFirstElement("VirtualProcessingUnits")->getUInt();
			unsigned int threadsPerVPU=0;

			if (_xmlData->getFirstElement("VirtualProcessingUnits")->findAttribute("ThreadsPerVPU")){
				threadsPerVPU=_xmlData->getFirstElement("VirtualProcessingUnits")->getAttributeAsUInt("ThreadsPerVPU");
			}
			cerr<<"updating VPU's numberOfVPUs="<<numberOfVPUs<<" threadsPerVPU="<<threadsPerVPU<<endl;
			pUtils->setVPUs(numberOfVPUs,threadsPerVPU);

			CC3DEventChangeNumberOfWorkNodes workNodeChangeEvent;
			workNodeChangeEvent.newNumberOfNodes=numberOfVPUs;

			// this will cause redundant calculations inside pUtils but since we do not call it often it is ok . This way code remains cleaner
			postEvent(workNodeChangeEvent);
		}

		if(_xmlData->getFirstElement("DebugOutputFrequency")){
			//updating DebugOutputFrequency in Potts using Metadata
			unsigned int debugOutputFrequency=_xmlData->getFirstElement("DebugOutputFrequency")->getUInt();
			potts.setDebugOutputFrequency(debugOutputFrequency>0 ?debugOutputFrequency: 0);
			ppdCC3DPtr->debugOutputFrequency=debugOutputFrequency;
		}

        CC3DXMLElementList npmVec=_xmlData->getElements("NonParallelModule");

        for (size_t i = 0 ; i<npmVec.size(); ++i){
            // this is simple initialization because for now we only allow Potts to have non-parallel execution. Adding more functionalty later will be straight-forward
            string moduleName=npmVec[i]->getAttribute("Name");
            if (moduleName=="Potts"){
                potts.setParallelUtils(pUtilsSingle);
            }
        }



}

void Simulator::initializeCC3D(){


	try{
		cerr<<"BEFORE initializePotts"<<endl;
		//initializePotts(ps.pottsParseData);
		initializePottsCC3D(ps.pottsCC3DXMLElement);



		//initializing parallel utils  - OpenMP
		pUtils->init(potts.getCellFieldG()->getDim());
        potts.setParallelUtils(pUtils); // by default Potts gets pUtls which can have multiple threads

        //initializing parallel utils  - OpenMP  - for single CPU runs of selecte modules
        pUtilsSingle->init(potts.getCellFieldG()->getDim());



		//after pUtils have been initialized we process metadata -  in this function potts may get pUtils limiting it to use single thread
		processMetadataCC3D(ps.metadataCC3DXMLElement);


		cerr<<"AFTER initializePotts"<<endl;
		std::set<std::string> initializedPlugins;
		std::set<std::string> initializedSteppables;

		for(size_t i=0; i <ps.pluginCC3DXMLElementVector.size(); ++i){

			std::string pluginName = ps.pluginCC3DXMLElementVector[i]->getAttribute("Name");
			bool pluginAlreadyRegisteredFlag=false;
			Plugin *plugin = pluginManager.get(pluginName,&pluginAlreadyRegisteredFlag);
			if(!pluginAlreadyRegisteredFlag){
				//Will only process first occurence of a given plugin
				cerr<<"INITIALIZING "<<pluginName<<endl;
				plugin->init(this, ps.pluginCC3DXMLElementVector[i]);
			}
		}

		for(size_t i=0; i <ps.steppableCC3DXMLElementVector.size(); ++i){
			std::string steppableName = ps.steppableCC3DXMLElementVector[i]->getAttribute("Type");
			bool steppableAlreadyRegisteredFlag=false;
			Steppable *steppable = steppableManager.get(steppableName,&steppableAlreadyRegisteredFlag);

			if(!steppableAlreadyRegisteredFlag){
				//Will only process first occurence of a given steppable
				cerr<<"INITIALIZING "<<steppableName<<endl;
				if(ps.steppableCC3DXMLElementVector[i]->findAttribute("Frequency"))
					steppable->frequency=ps.steppableCC3DXMLElementVector[i]->getAttributeAsUInt("Frequency");

				steppable->init(this, ps.steppableCC3DXMLElementVector[i]);
				classRegistry->addStepper(steppableName,steppable);

			}
		}
		if(ppdCC3DPtr->cellTypeMotilityVector.size()){
			//for(int i =0 ; i < ppdCC3DPtr->cellTypeMotilityVector.size() ;++i ){
			//	cerr<<" GOT THIS CELL TYPE FOR MOTILITY"<<	ppdCC3DPtr->cellTypeMotilityVector[i].typeName<<endl;
			//}

			potts.initializeCellTypeMotility(ppdCC3DPtr->cellTypeMotilityVector);
		}

	}catch (const BasicException &e) {
		cerr << "ERROR: " << e << endl;
		stringstream errorMessageStream;

		errorMessageStream<<"Exception during initialization/parsing :\n"<<e.getMessage()<<"\n"<<"Location \n"<<"FILE :"<<e.getLocation().getFilename()<<"\n"<<"LINE :"<<e.getLocation().getLine();
		recentErrorMessage=errorMessageStream.str();
		cerr<<"THIS IS recentErrorMessage="<<recentErrorMessage<<endl;
		if (!newPlayerFlag){
			throw e;
		}

	}
}




void Simulator::initializePottsCC3D(CC3DXMLElement * _xmlData){

	cerr<<"INSIDE initializePottsCC3D="<<endl;
	//registering Potts as SteerableObject
	registerSteerableObject(&potts);

	if (!ppdCC3DPtr){
		delete ppdCC3DPtr;
		ppdCC3DPtr=0;
	}

	ppdCC3DPtr= new PottsParseData();

	cerr<<"ppdCC3DPtr="<<ppdCC3DPtr<<"ppdCC3DPtr->dim="<<ppdCC3DPtr->dim<<endl;

	cerr<<"_xmlData->getFirstElement(Dimensions)->getAttributeAsUInt(x)="<<_xmlData->getFirstElement("Dimensions")->getAttributeAsUInt("x")<<endl;
	cerr<<"_xmlData->getFirstElement(Dimensions)->getAttributeAsUInt(y)="<<_xmlData->getFirstElement("Dimensions")->getAttributeAsUInt("y")<<endl;
	cerr<<"_xmlData->getFirstElement(Dimensions)->getAttributeAsUInt(z)="<<_xmlData->getFirstElement("Dimensions")->getAttributeAsUInt("z")<<endl;

	ppdCC3DPtr->dim.x = _xmlData->getFirstElement("Dimensions")->getAttributeAsUInt("x");
	ppdCC3DPtr->dim.y = _xmlData->getFirstElement("Dimensions")->getAttributeAsUInt("y");
	ppdCC3DPtr->dim.z = _xmlData->getFirstElement("Dimensions")->getAttributeAsUInt("z");


	//if (_xmlData->getFirstElement("FluctuationAmplitude")) {
	//	ppdCC3DPtr->temperature=_xmlData->getFirstElement("FluctuationAmplitude")->getDouble();
	//	fluctAmplGlobalReadFlag=true;
	//}


	bool fluctAmplGlobalReadFlag=false;

	bool fluctAmplByTypeReadFlag=false;
	if(_xmlData->getFirstElement("FluctuationAmplitude")){
		if (_xmlData->getFirstElement("FluctuationAmplitude")->findElement("FluctuationAmplitudeParameters")){

			CC3DXMLElementList motilityVec=_xmlData->getFirstElement("FluctuationAmplitude")->getElements("FluctuationAmplitudeParameters");
			for (size_t i = 0 ; i<motilityVec.size(); ++i){
				CellTypeMotilityData motilityData;

				motilityData.typeName=motilityVec[i]->getAttribute("CellType");
				motilityData.motility=static_cast<float>(motilityVec[i]->getAttributeAsDouble("FluctuationAmplitude"));
				ppdCC3DPtr->cellTypeMotilityVector.push_back(motilityData);
			}
			fluctAmplByTypeReadFlag=true;
		}else{
			ppdCC3DPtr->temperature=_xmlData->getFirstElement("FluctuationAmplitude")->getDouble();
			fluctAmplGlobalReadFlag=true;
		}

	}

	if (!fluctAmplGlobalReadFlag && _xmlData->getFirstElement("Temperature")) {
		ppdCC3DPtr->temperature=_xmlData->getFirstElement("Temperature")->getDouble();
	}


	if(!fluctAmplByTypeReadFlag && _xmlData->getFirstElement("CellMotility")){
		CC3DXMLElementList motilityVec=_xmlData->getFirstElement("CellMotility")->getElements("MotilityParameters");
		for (size_t i = 0 ; i<motilityVec.size(); ++i){
			CellTypeMotilityData motilityData;

			motilityData.typeName=motilityVec[i]->getAttribute("CellType");
			motilityData.motility=static_cast<float>(motilityVec[i]->getAttributeAsDouble("Motility"));
			ppdCC3DPtr->cellTypeMotilityVector.push_back(motilityData);
		}
	}

	if (_xmlData->getFirstElement("Steps")) {
		ppdCC3DPtr->numSteps=_xmlData->getFirstElement("Steps")->getUInt();
	}
	if (_xmlData->getFirstElement("Anneal")) {
		ppdCC3DPtr->anneal=_xmlData->getFirstElement("Anneal")->getUInt();
	}
	if (_xmlData->getFirstElement("Flip2DimRatio")) {
		ppdCC3DPtr->flip2DimRatio=_xmlData->getFirstElement("Flip2DimRatio")->getDouble();
	}





	ASSERT_OR_THROW("You must set Dimensions!", ppdCC3DPtr->dim.x!=0 || ppdCC3DPtr->dim.y!=0 || ppdCC3DPtr->dim.z!=0);
	potts.createCellField(ppdCC3DPtr->dim);
	//cerr<<"DIM="<<ppdCC3DPtr->dim<<endl;
	//cerr<<"Temp="<<_xmlData->getFirstElement("Temperature")->getDouble()<<endl;
	//cerr<<"Flip2DimRatio="<<_xmlData->getFirstElement("Flip2DimRatio")->getDouble()<<endl;

	std::string metropolisAlgorithmName="";
	if(_xmlData->getFirstElement("MetropolisAlgorithm"))
		metropolisAlgorithmName = _xmlData->getFirstElement("MetropolisAlgorithm")->getText();

	cerr << "_ppdCC3DPtr->algorithmName = " << metropolisAlgorithmName << endl;

	if(metropolisAlgorithmName!=""){
		potts.setMetropolisAlgorithm(metropolisAlgorithmName);
	}




	if(!_xmlData->getFirstElement("RandomSeed")){
		srand((unsigned int)time(0));
		unsigned int randomSeed=(unsigned int)rand()*((std::numeric_limits<unsigned int>::max)()-1);

		BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
		rand->setSeed(randomSeed);
	}else{
		BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
		rand->setSeed(_xmlData->getFirstElement("RandomSeed")->getUInt());
		ppdCC3DPtr->seed=_xmlData->getFirstElement("RandomSeed")->getUInt();
	}

	cerr << " ppdCC3DPtr->seed = " << ppdCC3DPtr->seed << endl;


	if (_xmlData->getFirstElement("Shape")) {
		ppdCC3DPtr->shapeFlag=true;
		ppdCC3DPtr->shapeAlgorithm = _xmlData->getFirstElement("Shape")->getAttribute("Algorithm");
		ppdCC3DPtr->shapeIndex = _xmlData->getFirstElement("Shape")->getAttributeAsInt("Index");
		ppdCC3DPtr->shapeSize = _xmlData->getFirstElement("Shape")->getAttributeAsInt("Size");;
		ppdCC3DPtr->shapeInputfile = _xmlData->getFirstElement("Shape")->getAttribute("File");;
		ppdCC3DPtr->shapeReg = _xmlData->getFirstElement("Shape")->getText();
	}


	if (_xmlData->getFirstElement("Boundary_x")) {
		ppdCC3DPtr->boundary_x = _xmlData->getFirstElement("Boundary_x")->getText();
	}
	if (_xmlData->getFirstElement("Boundary_y")) {
		ppdCC3DPtr->boundary_y = _xmlData->getFirstElement("Boundary_y")->getText();
	}

	if (_xmlData->getFirstElement("Boundary_z")) {
		ppdCC3DPtr->boundary_z = _xmlData->getFirstElement("Boundary_z")->getText();
	}

	//Initializing shapes - if used at all
	if(ppdCC3DPtr->shapeFlag){
		if (ppdCC3DPtr->shapeReg == "irregular") {
			ppdCC3DPtr->boundary_x = "noflux";
			ppdCC3DPtr->boundary_y = "noflux";
			ppdCC3DPtr->boundary_z = "noflux";
		}
	}
	//	cerr << "" <<  << endl;

	cerr << "ppdCC3DPtr->boundary_x = " << ppdCC3DPtr->boundary_x << endl;
	//setting boundary conditions
	if(ppdCC3DPtr->boundary_x!=""){
		potts.setBoundaryXName(ppdCC3DPtr->boundary_x);
	}

	cerr << "_ppdCC3DPtr->boundary_y = " << ppdCC3DPtr->boundary_y << endl;
	if(ppdCC3DPtr->boundary_y!=""){
		potts.setBoundaryYName(ppdCC3DPtr->boundary_y);
	}

	cerr << "ppdCC3DPtr->boundary_z = " << ppdCC3DPtr->boundary_z << endl;
	if(ppdCC3DPtr->boundary_z!=""){
		potts.setBoundaryZName(ppdCC3DPtr->boundary_z);
	}

	if (_xmlData->getFirstElement("LatticeType")) {
		ppdCC3DPtr->latticeType = _xmlData->getFirstElement("LatticeType")->getText();
	}

	cerr << "ppdCC3DPtr->latticeType = " << ppdCC3DPtr->latticeType << endl;



	changeToLower(ppdCC3DPtr->latticeType);



	BoundaryStrategy::destroy(); // TEMP, se what happens: It hangs after second selection of the file.





	// This is the ONLY place where the BoundaryStrategy singleton is instantiated!!!
	if(ppdCC3DPtr->latticeType=="hexagonal")
	{
		if(ppdCC3DPtr->boundary_x=="Periodic")
		{
			ASSERT_OR_THROW("For hexagonal lattice and x periodic boundary conditions x dimension must be an even number",!(ppdCC3DPtr->dim.x%2));
		}
		if(ppdCC3DPtr->boundary_y=="Periodic")
		{
			ASSERT_OR_THROW("For hexagonal lattice and y periodic boundary conditions y dimension must be an even number",!(ppdCC3DPtr->dim.y%2));
		}
		if(ppdCC3DPtr->boundary_z=="Periodic")
		{
			ASSERT_OR_THROW("For hexagonal lattice and z periodic boundary conditions z dimension must be a number divisible by 3",!(ppdCC3DPtr->dim.z%3));
		}

		BoundaryStrategy::instantiate(ppdCC3DPtr->boundary_x, ppdCC3DPtr->boundary_y, ppdCC3DPtr->boundary_z, ppdCC3DPtr->shapeAlgorithm, ppdCC3DPtr->shapeIndex, ppdCC3DPtr->shapeSize, ppdCC3DPtr->shapeInputfile,HEXAGONAL_LATTICE);
		cerr<<"initialized hex lattice"<<endl;
	}
	else
	{
		BoundaryStrategy::instantiate(ppdCC3DPtr->boundary_x, ppdCC3DPtr->boundary_y, ppdCC3DPtr->boundary_z, ppdCC3DPtr->shapeAlgorithm, ppdCC3DPtr->shapeIndex, ppdCC3DPtr->shapeSize, ppdCC3DPtr->shapeInputfile,SQUARE_LATTICE);
		cerr<<"initialized square lattice"<<endl;
	}

	cerr<<"potts.getLatticeType()="<<potts.getLatticeType()<<endl; //potts.getLatticeType() only works when the BoundaryStrategy singleton is instantiated!

	//    exit(0);
	BoundaryStrategy::getInstance()->setDim(ppdCC3DPtr->dim);

	if (_xmlData->getFirstElement("FlipNeighborMaxDistance")) {

		ppdCC3DPtr->depth=_xmlData->getFirstElement("FlipNeighborMaxDistance")->getDouble();
		ppdCC3DPtr->depthFlag=true;
	}

	if (_xmlData->getFirstElement("NeighborOrder")) {

		ppdCC3DPtr->neighborOrder=_xmlData->getFirstElement("NeighborOrder")->getUInt();
		ppdCC3DPtr->depthFlag=false;
	}

	if(ppdCC3DPtr->depthFlag)
	{
		potts.setDepth(ppdCC3DPtr->depth);
	}
	else
	{
		potts.setNeighborOrder(ppdCC3DPtr->neighborOrder);
	}

	cerr << "ppdCC3DPtr->depthFlag = " << ppdCC3DPtr->depthFlag << endl;

	if (_xmlData->getFirstElement("DebugOutputFrequency")) {
		ppdCC3DPtr->debugOutputFrequency=_xmlData->getFirstElement("DebugOutputFrequency")->getUInt();
	}


	cerr << "ppdCC3DPtr->debugOutputFrequency = " << ppdCC3DPtr->debugOutputFrequency << endl;
	if(ppdCC3DPtr->debugOutputFrequency<=0)
	{
		ppdCC3DPtr->debugOutputFrequency=0;
		potts.setDebugOutputFrequency(0);
	}
	else
	{
		potts.setDebugOutputFrequency(ppdCC3DPtr->debugOutputFrequency);
	}

	if (_xmlData->getFirstElement("AcceptanceFunctionName")) {
		ppdCC3DPtr->acceptanceFunctionName=_xmlData->getFirstElement("AcceptanceFunctionName")->getText();
	}
	//Setting Acceptance Function
	//    cerr<<"ppdCC3DPtr->acceptanceFunctionName="<<ppdCC3DPtr->acceptanceFunctionName<<endl;
	potts.setAcceptanceFunctionByName(ppdCC3DPtr->acceptanceFunctionName);
	//    exit(0);


	if (_xmlData->getFirstElement("Offset")) {//TODO: these two are really strange: assigning double into strings??
		ppdCC3DPtr->acceptanceFunctionName=_xmlData->getFirstElement("Offset")->getDouble();
	}

	if (_xmlData->getFirstElement("KBoltzman")) {
		ppdCC3DPtr->acceptanceFunctionName=_xmlData->getFirstElement("KBoltzman")->getDouble();
	}

	if(ppdCC3DPtr->offset!=0.)
	{
		potts.getAcceptanceFunction()->setOffset(ppdCC3DPtr->offset);
	}
	if(ppdCC3DPtr->kBoltzman!=1.0)
	{
		potts.getAcceptanceFunction()->setK(ppdCC3DPtr->kBoltzman);
	}

	if (_xmlData->getFirstElement("FluctuationAmplitudeFunctionName")) {
		ppdCC3DPtr->fluctuationAmplitudeFunctionName=_xmlData->getFirstElement("FluctuationAmplitudeFunctionName")->getText();
	}

	potts.setFluctuationAmplitudeFunctionByName(ppdCC3DPtr->fluctuationAmplitudeFunctionName);



	if(_xmlData->getFirstElement("EnergyFunctionCalculator"))
	{
		if(_xmlData->getFirstElement("EnergyFunctionCalculator")->findAttribute("Type")){
			std::string energyFunctionCalculatorType=_xmlData->getFirstElement("EnergyFunctionCalculator")->getAttribute("Type");
			if(energyFunctionCalculatorType=="Statistics"){
				potts.createEnergyFunction(energyFunctionCalculatorType);
			}
		}
		EnergyFunctionCalculator * enCalculator=potts.getEnergyFunctionCalculator();
		enCalculator->setSimulator(this);
		enCalculator->init(_xmlData->getFirstElement("EnergyFunctionCalculator"));
	}





	//Units
	//if(_xmlData->getFirstElement("Units")){
	//	CC3DXMLElement *unitElemPtr;
	//	unitElemPtr=_xmlData->getFirstElement("Units")->getFirstElement("MassUnit");
	//	if (unitElemPtr){
	//		ppdCC3DPtr->massUnit=Unit(unitElemPtr->getText());
	//		//potts.setMassUnit(ppdCC3DPtr->massUnit);
	//	}
	//	unitElemPtr=_xmlData->getFirstElement("Units")->getFirstElement("LengthUnit");
	//	if (unitElemPtr){
	//		ppdCC3DPtr->massUnit=Unit(unitElemPtr->getText());
	//		//potts.setLengthUnit(ppdCC3DPtr->lengthUnit);
	//	}
	//	unitElemPtr=_xmlData->getFirstElement("Units")->getFirstElement("TimeUnit");
	//	if (unitElemPtr){
	//		ppdCC3DPtr->massUnit=Unit(unitElemPtr->getText());
	//		//potts.setTimeUnit(ppdCC3DPtr->timeUnit);
	//	}

	//}

//	//set default values for units
//	potts.setMassUnit(ppdCC3DPtr->massUnit);
//	potts.setLengthUnit(ppdCC3DPtr->lengthUnit);
//	potts.setTimeUnit(ppdCC3DPtr->timeUnit);
//	//potts.setEnergyUnit(ppdCC3DPtr->massUnit*(ppdCC3DPtr->lengthUnit)*(ppdCC3DPtr->lengthUnit)/((ppdCC3DPtr->timeUnit)*(ppdCC3DPtr->timeUnit)));


	//this might reinitialize some of the POtts members but it also makes sure that units are initialized too.
	potts.update(_xmlData);
	cerr << "before return 1" << endl;

	return;
}


/////////////////////////////////// Steering //////////////////////////////////////////////
CC3DXMLElement * Simulator::getCC3DModuleData(std::string _moduleType,std::string _moduleName){
	if(_moduleType=="Potts"){
		return ps.pottsCC3DXMLElement;
	}else if(_moduleType=="Metadata"){
		return ps.metadataCC3DXMLElement;
	}else if(_moduleType=="Plugin"){
		for (size_t i = 0 ; i<ps.pluginCC3DXMLElementVector.size() ;  ++i){
			if (ps.pluginCC3DXMLElementVector[i]->getAttribute("Name")==_moduleName)
				return ps.pluginCC3DXMLElementVector[i];
		}
		return 0;
	}else if(_moduleType=="Steppable"){
		for (size_t i = 0 ; i<ps.pluginCC3DXMLElementVector.size() ;  ++i){
			if (ps.steppableCC3DXMLElementVector[i]->getAttribute("Type")==_moduleName)
				return ps.steppableCC3DXMLElementVector[i];
		}
		return 0;
	}else{
		return 0;
	}
}

void Simulator::updateCC3DModule(CC3DXMLElement *_element){
	if(!_element)
		return;
	if(_element->getName()=="Potts"){
		ps.updatePottsCC3DXMLElement=_element;
	}else if(_element->getName()=="Metadata"){
		ps.updateMetadataCC3DXMLElement=_element;
	}else if(_element->getName()=="Plugin"){
		ps.updatePluginCC3DXMLElementVector.push_back(_element);
	}else if(_element->getName()=="Steppable"){
		ps.updateSteppableCC3DXMLElementVector.push_back(_element);
	}
}

void Simulator::steer(){
	std::map<std::string,SteerableObject *>::iterator mitr;

	if(ps.updatePottsCC3DXMLElement){

		mitr=steerableObjectMap.find("Potts");
		if(mitr!=steerableObjectMap.end()){
			mitr->second->update(ps.updatePottsCC3DXMLElement);
			ps.pottsCC3DXMLElement=ps.updatePottsCC3DXMLElement;

			if(ps.updatePottsCC3DXMLElement->getFirstElement("Steps")){
				ppdCC3DPtr->numSteps=ps.updatePottsCC3DXMLElement->getFirstElement("Steps")->getUInt();
			}

			ps.updatePottsCC3DXMLElement=0;
		}

	}

	if(ps.updateMetadataCC3DXMLElement){
		//here we will update Debug Frequency - the way we handle this will have to be rewritten though. This is ad hoc solution only
		//mitr=steerableObjectMap.find("Potts");
		//if(mitr!=steerableObjectMap.end()){
		//	if(ps.updateMetadataCC3DXMLElement->getFirstElement("DebugOutputFrequency")){
		//		//updating DebugOutputFrequency in Potts using Metadata
		//		unsigned int debugOutputFrequency=ps.updateMetadataCC3DXMLElement->getFirstElement("DebugOutputFrequency")->getUInt();
		//		//((Potts3D*)(mitr->second))->setDebugOutputFrequency(debugOutputFrequency>0 ?debugOutputFrequency: 0);
		//		potts.setDebugOutputFrequency(debugOutputFrequency>0 ?debugOutputFrequency: 0);
		//		ppdCC3DPtr->debugOutputFrequency=debugOutputFrequency;
		//	}

		//}

		processMetadataCC3D(ps.updateMetadataCC3DXMLElement); // here we update number of work nodes and Debug output frequency
		ps.updateMetadataCC3DXMLElement=0;

	}


	else if(ps.updatePluginCC3DXMLElementVector.size()){

		string moduleName;
		for (size_t i = 0 ; i < ps.updatePluginCC3DXMLElementVector.size() ; ++i){
			moduleName=ps.updatePluginCC3DXMLElementVector[i]->getAttribute("Name");
			mitr=steerableObjectMap.find(moduleName);
			if(mitr!=steerableObjectMap.end()){
				mitr->second->update(ps.updatePluginCC3DXMLElementVector[i]);
				//now overwrite pointer to existing copy of the module data
				for(size_t j=0 ; j < ps.pluginCC3DXMLElementVector.size(); ++j){
					if(ps.pluginCC3DXMLElementVector[j]->getAttribute("Name")==moduleName)
						ps.pluginCC3DXMLElementVector[j]=ps.updatePluginCC3DXMLElementVector[i];
				}
			}

		}
		ps.updatePluginCC3DXMLElementVector.clear();
	}else if(ps.updateSteppableCC3DXMLElementVector.size()){

		string moduleName;
		for (size_t i = 0 ; i < ps.updateSteppableCC3DXMLElementVector.size() ; ++i){
			moduleName=ps.updateSteppableCC3DXMLElementVector[i]->getAttribute("Type");
			mitr=steerableObjectMap.find(moduleName);
			if(mitr!=steerableObjectMap.end()){
				mitr->second->update(ps.updateSteppableCC3DXMLElementVector[i]);

				//now overwrite pointer to existing copy of the module data
				for(size_t j=0 ; j < ps.steppableCC3DXMLElementVector.size(); ++j){
					if(ps.steppableCC3DXMLElementVector[j]->getAttribute("Type")==moduleName)
						ps.steppableCC3DXMLElementVector[j]=ps.updateSteppableCC3DXMLElementVector[i];
				}
			}
		}
		ps.updateSteppableCC3DXMLElementVector.clear();
	}

}
