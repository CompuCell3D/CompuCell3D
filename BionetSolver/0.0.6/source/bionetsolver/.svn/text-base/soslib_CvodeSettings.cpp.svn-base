
#include <iostream>
#include <sstream>

#include "bionetsolver/soslib_CvodeSettings.h"
#include "sbmlsolver/integratorSettings.h"

soslib_CvodeSettings::soslib_CvodeSettings() :
    nextTimeStep(1.0) {
    
    settings = NULL;
    std::cout << "New CvodeSettings object created, but not initialized..." << std::endl << std::endl;
    
    setEndTime(nextTimeStep);
}

soslib_CvodeSettings::soslib_CvodeSettings(double _integrationTime) :
    nextTimeStep(_integrationTime) {
    
    settings = NULL; std::stringstream ss;
    settings = CvodeSettings_createWithTime(nextTimeStep, 1);
    if( settings != NULL ) {
        //ss << "New CvodeSettings object created with time ";
        //ss << _integrationTime << std::endl;
        //std::cout << ss.str();
    } else {
        ss << "CvodeSettings creation unsuccessful. Null pointer for settings." << std::endl;
        std::cout << ss.str();
    }
    
    setEndTime(nextTimeStep);
}

soslib_CvodeSettings::soslib_CvodeSettings(double _integrationTime, unsigned int _numberOfSteps) :
    nextTimeStep(_integrationTime) {
    
    settings = NULL; std::stringstream ss;
    settings = CvodeSettings_createWithTime(nextTimeStep, _numberOfSteps);
    if( settings != NULL ) {
        //ss << "New CvodeSettings object created with time ";
        //ss << _integrationTime << " and with " << _numberOfSteps << " steps." << std::endl;
        //std::cout << ss.str();
    } else {
        ss << "CvodeSettings creation unsuccessful. Null pointer for settings." << std::endl;
        std::cout << ss.str();
    }
}

soslib_CvodeSettings* soslib_CvodeSettings::cloneSettings() const {
    soslib_CvodeSettings* newSettings = new soslib_CvodeSettings( getEndTime(), getPrintSteps() );
    return newSettings;
}

soslib_CvodeSettings::~soslib_CvodeSettings(){
    //std::cout << "Called soslib_CvodeSettings destructor." << std::endl;
    if( settings != NULL){
        //std::cout << "Calling CvodeSettings_free..." << std::endl;
        CvodeSettings_free(settings);
    }
}

void soslib_CvodeSettings::createSettings(){
    if (settings != NULL) CvodeSettings_free(settings);
    settings = CvodeSettings_create();
}

void soslib_CvodeSettings::setSettings(cvodeSettings_t * cvodeSettings){
    if (settings != NULL) CvodeSettings_free(settings);
    settings = cvodeSettings;
}

void soslib_CvodeSettings::createSettings(double _integrationTime, unsigned int _numberOfSteps){
    if (settings != NULL) CvodeSettings_free(settings);
    settings = CvodeSettings_createWithTime(_integrationTime, _numberOfSteps);
}

void soslib_CvodeSettings::printSettings(){
    if (settings != NULL){
        std::cout << "Number of integration steps: \t";
        std::cout << CvodeSettings_getPrintsteps(this->getSettings()) << std::endl;
        std::cout << "Integration time: \t";
        std::cout << CvodeSettings_getEndTime(this->getSettings()) << std::endl;
    }
}

cvodeSettings_t* soslib_CvodeSettings::getSettings() const {
    if (settings == NULL) 
        std::cout << "CvodeSettings has not been initialized." << std::endl;
    return settings;
}

double soslib_CvodeSettings::getTimeStep() const { 
    return nextTimeStep;
}

void soslib_CvodeSettings::setTimeStep(double timeValue) {
    nextTimeStep = timeValue;
}

double soslib_CvodeSettings::getEndTime() const {
    double endTime = 0.0;
    if( indefiniteIntegrationIsSet() ){
        endTime = settings->Time;
    } else {
        endTime = CvodeSettings_getEndTime(settings);
    }
    return endTime;
}

void soslib_CvodeSettings::setEndTime(double newEndTime){
    std::stringstream ss;
    //std::cout << "Attempting to set end time i.e. CvodeSettings_setTime( settings, newEndTime, getPrintSteps() )" << std::endl;
    //ss.str(""); ss << "New end time: " << newEndTime << std::endl; std::cout << ss.str();
    //ss.str(""); ss << "Print steps i.e. getPrintSteps(): " << getPrintSteps() << std::endl; std::cout << ss.str();
    CvodeSettings_setTime( settings, newEndTime, getPrintSteps() );
}

int soslib_CvodeSettings::getPrintSteps() const {
    int printSteps = 1;
    if( !indefiniteIntegrationIsSet() ){
        printSteps = CvodeSettings_getPrintsteps(settings);
    }
    return printSteps;
}

void soslib_CvodeSettings::setPrintSteps(int newPrintSteps){
    CvodeSettings_setTime( settings, getEndTime(), newPrintSteps );
}

void soslib_CvodeSettings::setIndefiniteIntegration(int indefinite){
    CvodeSettings_setIndefinitely(settings, indefinite);
}

int soslib_CvodeSettings::indefiniteIntegrationIsSet() const {
    return CvodeSettings_getIndefinitely(settings);
}




