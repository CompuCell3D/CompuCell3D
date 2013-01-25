
#include "bionetsolver/Bionetwork.h"
#include "bionetsolver/BionetworkTemplateLibrary.h"
#include "bionetsolver/BionetworkSBML.h"
#include "bionetsolver/soslib_IntegratorInstance.h"

#include "bionetsolver/BionetworkUtilManager.h"

#include <iostream>

using namespace std;

Bionetwork::Bionetwork() : 
    //divideVolume(65535),
    //initialVolume(0),
    utilManager(0) {
    
    //cellType.first = "";
    //cellType.second = 0;
    templateLibrary.first = "";
    templateLibrary.second = 0;
    
    utilManager = new BionetworkUtilManager();
}

//Bionetwork::Bionetwork(std::pair<std::string, const BionetworkTemplateLibrary *> _type) : 
Bionetwork::Bionetwork(std::pair<std::string, const BionetworkTemplateLibrary *> _template) : 
    //cellType(_type),
    templateLibrary(_template),
    //divideVolume(65535),
    //initialVolume(0),
    utilManager(0) {
    
    utilManager = new BionetworkUtilManager();
}


//Bionetwork::Bionetwork(std::string cellTypeName, const BionetworkTemplateLibrary * cellTypePtr) :
Bionetwork::Bionetwork(std::string templateLibraryName, const BionetworkTemplateLibrary * templateLibraryPtr) :
    //initialVolume(0),
    utilManager(0)
    {
    
    //cellType.first = cellTypeName;
    //cellType.second = cellTypePtr;
    templateLibrary.first = templateLibraryName;
    templateLibrary.second = templateLibraryPtr;
    //divideVolume = 65535;
    
    utilManager = new BionetworkUtilManager();
}


//Bionetwork::Bionetwork( Bionetwork * _cc3dsosCell ) :
Bionetwork::Bionetwork( Bionetwork * _bionet ) :
    utilManager(0) {
    
    //initialVolume = _cc3dsosCell->getInitialVolume();
    //divideVolume = _cc3dsosCell->getDivideVolume();
    
    utilManager = new BionetworkUtilManager();
    
    //cellType.first = _cc3dsosCell->getCellTypeName();
    //cellType.second = _cc3dsosCell->getCellTypeInstancePtr();
    templateLibrary.first = _bionet->getTemplateLibraryName();
    templateLibrary.second = _bionet->getTemplateLibraryInstancePtr();
    //initializeIntegrators( cellType.second->getSBMLModels() );
    initializeIntegrators( templateLibrary.second->getSBMLModels() );
    
    const soslib_IntegratorInstance* parentIntegrator;
    std::map<std::string, soslib_IntegratorInstance *>::iterator integrItr = integrators.begin();
    for(; integrItr != integrators.end(); ++integrItr){
        //parentIntegrator = _cc3dsosCell->getIntegrInstance(integrItr->first);
        parentIntegrator = _bionet->getIntegrInstance(integrItr->first);
        if( parentIntegrator == NULL ){
            std::cout << "WARNING: Got null pointer to parent bionetwork integrator." << std::endl;
        } else {
            integrItr->second->setState( parentIntegrator->getState() );
            integrItr->second->setParamValues( parentIntegrator->getParamValues() );
        }
    }
}

Bionetwork::~Bionetwork() {
    std::cout << "Bionetwork::~Bionetwork() was called. Cell is being destroyed.\n\n\n\n\n\n\n\n\n" << std::endl;
    std::map<std::string, soslib_IntegratorInstance *>::iterator itr;
    itr = integrators.begin();
    for(; itr != integrators.end(); ++itr){
        delete itr->second;
    }
    
    if( utilManager != 0){
        delete utilManager;
        utilManager = 0;
    }
}


//std::string Bionetwork::getCellTypeName() const {
std::string Bionetwork::getTemplateLibraryName() const {
    //std::string cellTypeName;
    //cellTypeName = cellType.first;
    //return cellTypeName;
    std::string templateLibraryName;
    templateLibraryName = templateLibrary.first;
    return templateLibraryName;
}


//std::string Bionetwork::cellTypeAsString() const {
std::string Bionetwork::templateLibraryAsString() const {
    //return cellType.first;
    return templateLibrary.first;
}

const Bionetwork * Bionetwork::getConstPointer() const {
    const Bionetwork * cellPointer = this;
    return cellPointer;
}


void Bionetwork::initializeIntegrators(){
    //initializeIntegrators( cellType.second->getSBMLModels() );
    initializeIntegrators( templateLibrary.second->getSBMLModels() );
}

void Bionetwork::initializeIntegrators(std::map<std::string, const BionetworkSBML *> models){
    
    soslib_IntegratorInstance* currentIntegrInstance;
    
    std::map<std::string, const BionetworkSBML *>::const_iterator modelItr = models.begin();
    for(; modelItr != models.end(); ++modelItr){
        currentIntegrInstance = new soslib_IntegratorInstance(
            modelItr->second->getOdeModel(), modelItr->second->getSettings() );
        integrators[modelItr->first] = currentIntegrInstance;
        integrators[modelItr->first]->setModelKey( modelItr->second->getModelKey() );
        integrators[modelItr->first]->setModelName( modelItr->first );
    }
    
    //this->setIntracellularState(
    this->setBionetworkState(
		//cellType.second->getInitialConditions("SBML") );
        templateLibrary.second->getInitialConditions() );
}


void Bionetwork::addNewIntegrator(const BionetworkSBML * model){
    soslib_IntegratorInstance* newIntegrInstance;
    soslib_CvodeSettings* newSettings = 
        new soslib_CvodeSettings(model->getTimeStepSize(), 1);
    newIntegrInstance = new soslib_IntegratorInstance( model->getOdeModel(), newSettings );
    //newIntegrInstance->setState( cellType.second->getInitialConditions("SBML") );
    newIntegrInstance->setState( templateLibrary.second->getInitialConditions() );
    integrators[model->getModelName()] = newIntegrInstance;
}


//void Bionetwork::setIntracellularState(std::map<std::string, double> state){ 
void Bionetwork::setBionetworkState(std::map<std::string, double> state){ 
    std::map<std::string, soslib_IntegratorInstance *>::iterator it = integrators.begin();
    for(; it != integrators.end(); ++it){
        it->second->setState(state);
    }
}


std::vector<std::string> Bionetwork::getSBMLModelNames() const {
    std::vector<std::string> sbmlModelNames;
    std::map<std::string, soslib_IntegratorInstance *>::const_iterator integrItr = integrators.begin();
    for(; integrItr != integrators.end(); ++integrItr){
        sbmlModelNames.push_back(integrItr->first);
    }
    return sbmlModelNames;
}

string Bionetwork::getSBMLModelNamesAsString() const{
	string sbmlModelNames;

    std::map<std::string, soslib_IntegratorInstance *>::const_iterator integrItr = integrators.begin();
    for(; integrItr != integrators.end(); ++integrItr){
        sbmlModelNames+=integrItr->first+" ";
    }
    return sbmlModelNames;

}



bool Bionetwork::hasSBMLModelByKey(std::string modelKey) const {
    //return cellType.second->hasSBMLModelByKey(modelKey);
    return templateLibrary.second->hasSBMLModelByKey(modelKey);
}


//void Bionetwork::setCellType(const BionetworkTemplateLibrary *type ) {
//        cellType.first = type->getTypeName();
//        cellType.second = type;
//}
void Bionetwork::setTemplateLibrary(const BionetworkTemplateLibrary *newTemplateLibrary ) {
        templateLibrary.first = newTemplateLibrary->getTemplateLibraryName();
        templateLibrary.second = newTemplateLibrary;
}

//void Bionetwork::changeCellType(const BionetworkTemplateLibrary *type ) {
void Bionetwork::changeTemplateLibrary(const BionetworkTemplateLibrary *newTemplateLibrary ) {
//void           changeTemplateLibrary(const BionetworkTemplateLibrary * );
        //cellType.first = type->getTypeName();
        //cellType.second = type;
        templateLibrary.first = newTemplateLibrary->getTemplateLibraryName();
        templateLibrary.second = newTemplateLibrary;
        std::map<std::string, soslib_IntegratorInstance *>::iterator integrItr = integrators.begin();
        for(; integrItr != integrators.end(); ++integrItr){
            //const BionetworkSBML * sbmlModel = type->getSBMLModelByName(integrItr->first);
            const BionetworkSBML * sbmlModel = newTemplateLibrary->getSBMLModelByName(integrItr->first);
            if( sbmlModel == NULL){
                delete integrItr->second;
                integrItr->second = NULL;
                integrators.erase(integrItr);
            }
        }
        
        //std::map<std::string, const BionetworkSBML *> models = type->getSBMLModels();
        std::map<std::string, const BionetworkSBML *> models = newTemplateLibrary->getSBMLModels();
        std::map<std::string, const BionetworkSBML *>::const_iterator modelItr = models.begin();
        for(; modelItr != models.end(); ++modelItr){
            integrItr = integrators.find(modelItr->first);
            if( integrItr == integrators.end() ){
                addNewIntegrator(modelItr->second);
            }
        }
}

//void Bionetwork::printIntracellularState() const {
void Bionetwork::printBionetworkState() const {
    std::map<std::string, soslib_IntegratorInstance*>::const_iterator it = integrators.begin();
    for(; it != integrators.end(); ++it){
        std::cout << "Current state of integrator for model " << it->first << std::endl;
        std::cout << it->second->getStateAsString() << std::endl;
    }
}

//void Bionetwork::printIntracellularState(bool printWithStateVarNames) const {
void Bionetwork::printBionetworkState(bool printWithStateVarNames) const {
    std::map<std::string, soslib_IntegratorInstance*>::const_iterator it = integrators.begin();
    for(; it != integrators.end(); ++it){
        std::cout << "Current state of integrator for model " << it->first << std::endl;
        pair<std::string, std::string> currentState = it->second->getStateAsString(false);
        if(printWithStateVarNames)
            std::cout << currentState.first << std::endl;
        std::cout << currentState.second << std::endl;
    }
}

//std::string Bionetwork::getIntracellStateVarNamesAsString( std::string modelName ) const {
std::string Bionetwork::getBionetworkStateVarNamesAsString( std::string modelName ) const {
    std::stringstream stateString;
    stateString << "";
    
    std::map<std::string, soslib_IntegratorInstance*>::const_iterator it = integrators.begin();
    for(; it != integrators.end(); ++it){
        if( it->first == modelName ) {
            pair<std::string, std::string> currentState = it->second->getStateAsString(false);
            stateString << currentState.first << std::endl;
            break;
        }
    }
    
    return stateString.str();
}

//std::string Bionetwork::getIntracellStateAsString( std::string modelName ) const {
std::string Bionetwork::getBionetworkStateAsString( std::string modelName ) const {
    std::stringstream stateString;
    stateString << "";
    
    std::map<std::string, soslib_IntegratorInstance*>::const_iterator it = integrators.begin();
    for(; it != integrators.end(); ++it){
        if( it->first == modelName ) {
            pair<std::string, std::string> currentState = it->second->getStateAsString(false);
            stateString << currentState.second << std::endl;
            break;
        }
    }
    
    return stateString.str();
}

//std::string Bionetwork::getIntracellularStateAsString( bool withStateVariableNames ) const {
std::string Bionetwork::getBionetworkStateAsString( bool withStateVariableNames ) const {
    std::stringstream stateString;
    
    std::map<std::string, soslib_IntegratorInstance*>::const_iterator it = integrators.begin();
    for(; it != integrators.end(); ++it){
        pair<std::string, std::string> currentState = it->second->getStateAsString(false);
        
        if( withStateVariableNames )
            stateString << currentState.first << std::endl;
        stateString << currentState.second << std::endl;
    }
    
    return stateString.str();
}


//void Bionetwork::updateIntracellularState(){
void Bionetwork::updateBionetworkState(){
    std::map<std::string, soslib_IntegratorInstance*>::iterator it = integrators.begin();
    for(; it != integrators.end(); ++it){
        it->second->integrateOneStep();
    }
}

//void Bionetwork::updateIntracellularStateWithTimeStep( double timeStep ){
void Bionetwork::updateBionetworkStateWithTimeStep( double timeStep ){
    std::stringstream ss;
    std::cout << "Bionetwork::updateBionetworkStateWithTimeStep called..." << std::endl;
    std::map<std::string, soslib_IntegratorInstance*>::iterator it = integrators.begin();
    for(; it != integrators.end(); ++it){
        if( !(it->second->indefiniteIntegrationIsSet()) ){
            std::cout << "Setting indefinite integration..." << std::endl;
            it->second->setIndefiniteIntegration(1);
        }
        ss.str(""); ss << "Setting next time step of integration to " << timeStep << std::endl;
        std::cout << ss.str();
        it->second->setNextTimeStep( timeStep );
        
        std::cout << "Integrating for one time step..." << std::endl;
        it->second->integrateOneStep();
    }
}

const soslib_IntegratorInstance* Bionetwork::getIntegrInstance(std::string modelName) const {
    const soslib_IntegratorInstance *integrInstance = NULL;
    std::map<std::string, soslib_IntegratorInstance *>::const_iterator itr;
    for(itr = integrators.begin(); itr != integrators.end(); ++itr){
        if (itr->first == modelName)
            integrInstance = itr->second;
    }
    return integrInstance;
}

std::map<std::string, const soslib_IntegratorInstance *> Bionetwork::getIntegrInstances() const {
    std::map<std::string, const soslib_IntegratorInstance *> returnIntegrInstances;
    std::map<std::string, soslib_IntegratorInstance *>::const_iterator itr;
    for(itr = integrators.begin(); itr != integrators.end(); ++itr)
        returnIntegrInstances[itr->first] = itr->second;
    return returnIntegrInstances;
}

void Bionetwork::printMessage() const {
    std::cout << "This is the print message function." << std::endl;
}

std::map<std::string, double> Bionetwork::getBionetworkParams(const std::string & modelName) const {
//     std::map<std::string, soslib_IntegratorInstance *>::const_iterator integrItr = integrators.begin();
    
    std::map<std::string, soslib_IntegratorInstance *>::const_iterator integrItr=integrators.find(modelName);
    if (integrItr==integrators.end()){
        integrItr = integrators.begin();
    }
    
    
    return integrItr->second->getParamValues();
}

std::map<std::string, double> Bionetwork::getBionetworkState(const std::string & modelName) const {
    
    std::map<std::string, soslib_IntegratorInstance *>::const_iterator integrItr=integrators.find(modelName);
    if (integrItr==integrators.end()){
        integrItr = integrators.begin();
    }
    
//     std::map<std::string, soslib_IntegratorInstance *>::const_iterator integrItr = integrators.begin();
    return integrItr->second->getState();
}


std::pair<bool, double> Bionetwork::findPropertyValue( std::string property ) const {

    bool valueFound = false;
    std::pair<bool, double> value;
    std::map<std::string, soslib_IntegratorInstance *>::const_iterator integrItr = integrators.begin();
    
    for(; integrItr != integrators.end(); ++integrItr){
//         cerr<<"integrator name="<<integrItr->first<<endl;
        value = integrItr->second->findValueAsDouble( property );
//         value =  std::pair<bool, double>(true,1.0);
        if( value.first ){
            valueFound = true;
            break;
        }
    }
    
    return value;
}


void Bionetwork::setPropertyValue( std::string property, double value ) {

    //bool cc3dNamespaceFound = false;
    //std::pair<std::string, double> nameValuePair;
    //std::pair<std::string, std::string> splitString;
    //if( utilManager->charFoundInString('_', property) ){
    //    splitString = utilManager->splitStringAtFirst( '_', property );
    //    //if( splitString.first == "cc3d" ){
    //    //    cc3dNamespaceFound = true;
    //    //} else if(splitString.first == "GGH"){
    //    //    cc3dNamespaceFound = true;
    //    //} else {
    //    //    cc3dNamespaceFound = false; 
    //    //}
    //} 
    
    //if( !cc3dNamespaceFound ){
    //    std::map<std::string, soslib_IntegratorInstance *>::iterator integrItr = integrators.begin();
    //    for(; integrItr != integrators.end(); ++integrItr){
    //        integrItr->second->setStateValue( property, value );
    //    }
    //}
    
    std::map<std::string, soslib_IntegratorInstance *>::iterator integrItr = integrators.begin();
    for(; integrItr != integrators.end(); ++integrItr){
        integrItr->second->setStateValue( property, value );
    }
    
}

//void Bionetwork::setCellPropertyValues( 
//    std::map<std::string, double> properties ){
//    
//    std::map<std::string, double>::iterator propItr = properties.begin();
//    for(; propItr != properties.end(); ++propItr){
//        setCellPropertyValue(*propItr );
//    }
//}

//void Bionetwork::setCellPropertyValue(
//    std::pair<std::string, double> propValue ){
//    
//    if( propValue.first == "DivideVolume" ){
//        setDivideVolume(propValue.second);
//    }
//    if( propValue.first == "InitialVolume" ){
//        setInitialVolume(propValue.second);
//    }
//}


//std::pair<std::string, double> 
//    Bionetwork::getCellPropertyAsPair(std::string _property ) const {
//    
//    std::pair<std::string, double> returnPair;
//    returnPair.first = _property;
//    returnPair.second = getCellPropertyAsDouble(_property );
//    return returnPair;
//}

//double Bionetwork::getCellPropertyAsDouble(std::string _property ) const {
//    double returnValue = 0.0;
//    
//    if( _property == "DivideVolume" ){
//        returnValue = getDivideVolume();
//    }
//    if( _property == "InitialVolume" ){
//        returnValue = getInitialVolume();
//    }
//    
//    return returnValue;
//}


//std::pair<bool, double> Bionetwork::findCellPropertyValue( std::string _property ) const {
//    
//    bool valueFound = false;
//    double returnValue = 0.0;
//    std::pair<bool, double> value;
//    
//    if( _property == "DivideVolume" ){
//        returnValue = getDivideVolume();
//    }
//    if( _property == "InitialVolume" ){
//        returnValue = getInitialVolume();
//    }
//    
//    value.first = valueFound;
//    value.second = returnValue;
//    return value;
//}










