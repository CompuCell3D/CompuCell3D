
#include <iostream>
#include <sstream>
#include "bionetsolver/soslib_IntegratorInstance.h"
//#include "cc3dsoslib/cc3dsos_UtilitiesManager.h"
#include "bionetsolver/BionetworkUtilManager.h"


soslib_IntegratorInstance::soslib_IntegratorInstance() :
    //newTimeStepHasBeenSet(true),
    ii(0),
    odeModel(0),
    settings(0),
    utilManager(0) {
    
    //ii = NULL; odeModel = NULL; settings = NULL; utilManager = NULL;
    //initialTimeStep = -1.0;
    //utilManager = new cc3dsos_UtilitiesManager();
    utilManager = new BionetworkUtilManager();
}

//soslib_IntegratorInstance::soslib_IntegratorInstance(
//    const soslib_OdeModel* _om, soslib_CvodeSettings* _settings){
soslib_IntegratorInstance::soslib_IntegratorInstance(
    const soslib_OdeModel* _om, const soslib_CvodeSettings* inputSettings) :
    //newTimeStepHasBeenSet(true),
    ii(0),
    odeModel(0),
    settings(0),
    utilManager(0) {
    
    //ii = NULL; odeModel = NULL; settings = NULL; utilManager = NULL;
    createIntegratorInstance(_om, inputSettings);
    
    //if( settings != NULL){
    //    initialEndTime = settings->getTimeStep();
    //} else {
    //    std::cout << "WARNING: Null soslib_CvodeSettings* even though 'createIntegratorInstance' was called." << std::endl;
    //    initialEndTime = -1.0;
    //}
    //utilManager = new cc3dsos_UtilitiesManager();
	utilManager = new BionetworkUtilManager();
}

soslib_IntegratorInstance::~soslib_IntegratorInstance(){
    if( utilManager != 0){
        delete utilManager;
        utilManager = 0;
    }
    if( settings != 0 ){
        delete settings;
        settings = 0;
    }
}

//void soslib_IntegratorInstance::createIntegratorInstance(
//    const soslib_OdeModel* _odeModel, soslib_CvodeSettings* _cvodeSettings){
void soslib_IntegratorInstance::createIntegratorInstance(
    const soslib_OdeModel* _odeModel, const soslib_CvodeSettings* inputSettings){
    
    if (ii != NULL){
        //IntegratorInstance_free(ii);
        ii = NULL;
    }
    odeModel = _odeModel;
    
    settings = inputSettings->cloneSettings();
    
    //settings->setIndefiniteIntegration(1);
    
    ii = IntegratorInstance_create(
        odeModel->getOdeModel(), settings->getSettings());
    //std::cout << "New soslib_IntegratorInstance created..." << std::endl;
}

std::map<std::string, double> soslib_IntegratorInstance::getState() const {
    std::map<std::string, double> state;
    if (ii != NULL){
        std::vector<variableIndex_t *> varIndexes(odeModel->getStateVariableIndexes());
        for (std::vector<variableIndex_t *>::iterator it = varIndexes.begin();
            it != varIndexes.end(); ++it){
            if( *it != NULL ){
                state[VariableIndex_getName(*it, odeModel->getOdeModel())] = 
                    IntegratorInstance_getVariableValue(ii, *it);
                VariableIndex_free(*it);
                *it = NULL;
            }
        }
    }
    return state;
}

void soslib_IntegratorInstance::setStateValue(std::string name, double value){
    std::pair<std::string, double> nameValuePair;
    nameValuePair.first = name;
    nameValuePair.second = value;
    setStateValue( nameValuePair );
}

void soslib_IntegratorInstance::setStateValue(std::pair<std::string, double> nameValuePair){
    std::map<std::string, double> stateMap;
    stateMap[nameValuePair.first] = nameValuePair.second;
    setState( stateMap );
}

void soslib_IntegratorInstance::setState(std::map<std::string, double> state){
    if (ii != NULL){
        //std::cout << "soslib_IntegratorInstance::setState was called..." << std::endl;
        //std::cout << "Size of map provided as input argument: " << state.size() << std::endl;
        variableIndex_t* currentVariableIndex = NULL;
        for (std::map<std::string, double>::iterator it = state.begin();
            it != state.end(); ++it){
            
            std::pair<std::string, std::string> splitString;
            splitString = utilManager->splitStringAtFirst('_', it->first);
            
            bool namespaceFound = false;
            if( utilManager->charFoundInString('_', it->first) ){
                namespaceFound = true;
            }
            
            //std::cout << "Inside 'for' loop inside of soslib_IntegratorInstance::setState..." << std::endl;
            //currentVariableIndex = NULL;
            if( namespaceFound ){
                //if( splitString.first == getModelKey() ){ 
                if( (splitString.first == getModelKey()) || (splitString.first == getModelName()) ){ 
                    currentVariableIndex = 
                        ODEModel_getVariableIndex( odeModel->getOdeModel(), splitString.second.c_str() );
                }
            } else {
                currentVariableIndex = 
                    ODEModel_getVariableIndex( odeModel->getOdeModel(), it->first.c_str() );
            }
            
            // Note if a namespace is found and it does not match the integrator model key, 
            //   currentVariableIndex will remain NULL and the following will not be executed.
            if( currentVariableIndex != NULL ){
                //std::cout << "Calling IntegratorInstance_setVariableValue..." << std::endl;
                IntegratorInstance_setVariableValue( ii, currentVariableIndex, it->second );
                //std::cout << "Just called IntegratorInstance_setVariableValue..." << std::endl;
                VariableIndex_free(currentVariableIndex);
                currentVariableIndex = NULL;
            }
        }
    }
}

std::map<std::string, double> soslib_IntegratorInstance::getParamValues() const {
    std::map<std::string, double> paramVals;
    if (ii != NULL){
        std::vector<variableIndex_t *> paramIndexes(odeModel->getParameterVariableIndexes());
        for (std::vector<variableIndex_t *>::iterator it = paramIndexes.begin();
            it != paramIndexes.end(); ++it){
            
            if( *it != NULL){
                paramVals[VariableIndex_getName(*it, odeModel->getOdeModel())] = 
                    IntegratorInstance_getVariableValue(ii, *it);
                VariableIndex_free(*it);
                *it = NULL;
            }
        }
    }
    return paramVals;
}


std::pair<bool, double> soslib_IntegratorInstance::findValueAsDouble(std::string valueName) const {
    bool valueFound = false;
    double returnValue = 0.0;
    
    if( ii != NULL ){
        bool namespaceFound = false;
        variableIndex_t* varIndex = NULL;
        std::pair<std::string, std::string> splitString;
        
        if( utilManager->charFoundInString('_', valueName) ){
            namespaceFound = true;
            splitString = utilManager->splitStringAtFirst('_', valueName);
        }
        
        if( namespaceFound ){
            //if( splitString.first == getModelKey() ){ 
            if( (splitString.first == getModelKey()) || (splitString.first == getModelName()) ){
                varIndex = 
                    ODEModel_getVariableIndex( odeModel->getOdeModel(), splitString.second.c_str() );
            }
        } else {
            varIndex = 
                ODEModel_getVariableIndex( odeModel->getOdeModel(), valueName.c_str() );
        }
        
        if( varIndex != NULL ){
            valueFound = true;
            returnValue = IntegratorInstance_getVariableValue( ii, varIndex );
            VariableIndex_free(varIndex);
            varIndex = NULL;
        }
    }
    
    std::pair<bool, double> returnPair;
    returnPair.first = valueFound;
    returnPair.second = returnValue;
    return returnPair;
}



double soslib_IntegratorInstance::getValueAsDouble(std::string valueName) const {
    double returnValue = 0.0;
    variableIndex_t* varIndex = NULL;
    if (ii != NULL){
        //VariableIndex_free(currentVariableIndex);
        //currentVariableIndex = NULL;
        varIndex = odeModel->getVariableIndex(valueName);
        //returnValue = IntegratorInstance_getVariableValue(
        //    ii, odeModel->getVariableIndex(valueName) );
        if( varIndex != NULL ){
            returnValue = IntegratorInstance_getVariableValue( ii, varIndex );
            VariableIndex_free(varIndex);
            varIndex = NULL;
        }
    }
    return returnValue;
}

void soslib_IntegratorInstance::setParamValues(std::map<std::string, double> paramValues){
    if (ii != NULL){
        variableIndex_t* varIndex = NULL;
        for (std::map<std::string, double>::iterator it = paramValues.begin();
            it != paramValues.end(); ++it){
            
            varIndex = ODEModel_getVariableIndex(odeModel->getOdeModel(), it->first.c_str());
            //IntegratorInstance_setVariableValue(ii,
            //    ODEModel_getVariableIndex(odeModel->getOdeModel(), it->first.c_str()),
            //    it->second);
            if( varIndex != NULL ){
                IntegratorInstance_setVariableValue(ii, varIndex, it->second);
                VariableIndex_free(varIndex);
                varIndex = NULL;
            }
        }
    }
}

std::string soslib_IntegratorInstance::getStateAsString(){
    std::ostringstream stateString("");
    if (ii != NULL){
        std::map<std::string, double> state = this->getState();
        
        stateString << IntegratorInstance_getTime(ii);
        for(std::map<std::string, double>::iterator it = state.begin();
            it != state.end(); ++it){
            stateString << "\t";
            stateString << it->second;
        }
    }
    return stateString.str();
}

pair<std::string, std::string> soslib_IntegratorInstance::getStateAsString(bool withTime){
    std::ostringstream stateVariableNameString("");
    std::ostringstream stateVariableValueString("");
    if (ii != NULL){
        std::map<std::string, double> state = this->getState();
        if(withTime){
            stateVariableNameString << "\t";
            stateVariableValueString << IntegratorInstance_getTime(ii) << "\t";
        }
        for(std::map<std::string, double>::iterator it = state.begin();
            it != state.end(); ++it){
            if(it != state.begin()){
                stateVariableNameString << "\t";
                stateVariableValueString << "\t";
            }
            stateVariableNameString << it->first;
            stateVariableValueString << it->second;
        }
    }
    return std::pair<std::string, std::string>
        (stateVariableNameString.str(), stateVariableValueString.str());
}

pair<std::string, std::string> soslib_IntegratorInstance::getParamValuesAsString(bool withTime){
    std::ostringstream paramNameString("");
    std::ostringstream paramValueString("");
    if (ii != NULL){//std::map<std::string, double> getParamValues()
        std::map<std::string, double> paramValues = this->getParamValues();
        if(withTime){
            paramNameString << "\t";
            paramValueString << IntegratorInstance_getTime(ii) << "\t";
        }
        for(std::map<std::string, double>::iterator it = paramValues.begin();
            it != paramValues.end(); ++it){
            if(it != paramValues.begin()){
                paramNameString << "\t";
                paramValueString << "\t";
            }
            paramNameString << it->first;
            paramValueString << it->second;
        }
    }
    return std::pair<std::string, std::string>
        (paramNameString.str(), paramValueString.str());
}

void soslib_IntegratorInstance::printIntegrationResults(){
    if (ii != NULL){
        std::cout << "\t" << odeModel->getStateVariablesAsString() << std::endl;
        std::cout << this->getStateAsString() << std::endl;
        for (int i=0; i< CvodeSettings_getPrintsteps(settings->getSettings()); i++){
            IntegratorInstance_integrateOneStep(ii);
            std::cout << this->getStateAsString() << std::endl;
        }
    }
}

void soslib_IntegratorInstance::resetIntegrator(){
    if (ii != NULL){
        IntegratorInstance_reset(ii);
    }
}

void soslib_IntegratorInstance::setIntegrator( const soslib_CvodeSettings * newSettings ) {
    if (ii != NULL){
        IntegratorInstance_set( ii, newSettings->getSettings() );
    }
}

//bool soslib_IntegratorInstance::validInitialTimeStep() const {
//    bool validInitialTimeStep = false;
//    if( initialEndTime > 0.0 ){
//        validInitialTimeStep = true;
//    }
//    return validInitialTimeStep;
//}

/*
void setNextTimeStep( double );
double getNextTimeStep() const ;
void setCurrentEndTime();
double getCurrentEndTime() const { return currentEndTime; };
*/

bool soslib_IntegratorInstance::indefiniteIntegrationIsSet() const {
    bool isSetToIndefiniteIntegration = false;
    if( settings->indefiniteIntegrationIsSet() != 0 ){
        isSetToIndefiniteIntegration = true;
    }
    return isSetToIndefiniteIntegration;
}

void soslib_IntegratorInstance::setIndefiniteIntegration( int indefiniteIntegrationSetting ) {
    settings->setIndefiniteIntegration(indefiniteIntegrationSetting);
}

void soslib_IntegratorInstance::setNextTimeStep( double newTimeStep ) {
    if( settings != NULL ){
        std::stringstream ss;
        settings->setTimeStep(newTimeStep);
        //newTimeStepHasBeenSet = true;
        ss.str("");
    } else {
        std::cout << "\n** WARNING: Attempted to set time step for Null integrator instance (" << getModelName() << ")" << std::endl;
        std::cout << "-- Will not set a new time step of integration.\n" << std::endl;
    }
}

double soslib_IntegratorInstance::getNextTimeStep() const {
    double timeStep = 0.0;
    if( settings != NULL ){
        timeStep = settings->getTimeStep();
    }
    else{
        std::cout << "\n** WARNING: Attempted to get time step for Null integrator instance (" << getModelName() << ")" << std::endl;
        std::cout << "-- Returning a value of 0.0 for time step of integration.\n" << std::endl;
    }
    return timeStep;
}

void soslib_IntegratorInstance::setCurrentEndTime(double newEndTime) {
    if( settings != NULL ){
        settings->setEndTime(newEndTime);
    } else {
        std::cout << "\n** WARNING: Attempted to set end time for NULL integrator instance (" << getModelName() << ")" << std::endl;
        std::cout << "-- Will not set a new end time of integration" << std::endl;
    }
}

double soslib_IntegratorInstance::getCurrentEndTime() const { 
    double endTime = 0.0;
    if( settings != NULL ){
        endTime = settings->getEndTime();
    } else {
        std::cout << "\n** WARNING: Attempted to get end time for NULL integrator instance (" << getModelName() << ")" << std::endl;
        std::cout << "-- Returning a value of 0.0 for end time of integration\n" << std::endl;
    }
    return endTime; 
}

void soslib_IntegratorInstance::integrateOneStep(){
    std::stringstream ss;
    if (ii != NULL){
        if( settings != NULL ){
            //std::cout << "Checking to see if indefinite integration is set..." << std::endl;
            if( settings->indefiniteIntegrationIsSet() ){
                
                std::cout << "Calculating: newEndTime = settings->getEndTime() + settings->getTimeStep()" << std::endl;
                double newEndTime = settings->getEndTime() + settings->getTimeStep();
                
                ss.str(""); ss << "Here's the old end time: " << settings->getEndTime(); std::cout << ss.str() << std::endl;
                ss.str(""); ss << "Here's the current time step: " << settings->getTimeStep(); std::cout << ss.str() << std::endl;
                ss.str(""); ss << "Here's the new end time: " << newEndTime; std::cout << ss.str() << std::endl;
                
                //std::cout << "Setting new end time: settings->setEndTime(newEndTime)" << std::endl;
                settings->setEndTime(newEndTime);
                
                //std::cout << "Test to see if settings->getEndTime() works..." << std::endl;
                double theEndTime = settings->getEndTime();
                //ss.str(""); ss << "Here's the end time: " << theEndTime << std::endl; std::cout << ss.str();
                
                //ss.str(""); ss << "-- New integrator time step: " << settings->getEndTime();
                std::cout << ss.str() << std::endl;
                //std::cout << "Calling IntegratorInstance_setNextTimeStep( ii, settings->getEndTime() )..." << std::endl;
                IntegratorInstance_setNextTimeStep( ii, settings->getEndTime() );
                //std::cout << "Calling IntegratorInstance_integrateOneStep(ii)..." << std::endl;
                IntegratorInstance_integrateOneStep(ii);
            } else {
                integrateOneStepAndResetIntegrator();
            }
        } else {
            std::cout << "\n** WARNING: NULL settings for integrator instance (" << getModelName() << ")" << std::endl;
            std::cout << "-- Cannot set next integration time step.\n" << std::endl;
        }
    } else {
        std::cout << "\n** WARNING: NULL integrator instance (" << getModelName() << ")" << std::endl;
        std::cout << "-- Cannot time-step integrator.\n" << std::endl;
    }
}

void soslib_IntegratorInstance::integrateOneStepAndResetIntegrator(){

    if (ii != NULL){
        IntegratorInstance_integrateOneStep(ii);
        std::map<std::string, double> currentState = getState();
        std::map<std::string, double> currentParamValues = getParamValues();
        
        resetIntegrator();
        setState(currentState);
        setParamValues(currentParamValues);
        
    } else {
        std::cout << "\nWARNING: Invalid soslib_IntegratorInstance (" << getModelName() << ")" << std::endl;
        std::cout << "Current ii (soslib integrator instance) has a NULL value.";
        std::cout << " It appears that this integrator instance has NOT been initialized." << std::endl;
        std::cout << "Will not integrate for one time step.\n" << std::endl;
    }
}





