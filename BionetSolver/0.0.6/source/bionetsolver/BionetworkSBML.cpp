


#include "bionetsolver/BionetworkSBML.h"
#include "bionetsolver/soslib_OdeModel.h"
#include "bionetsolver/soslib_CvodeSettings.h"

BionetworkSBML::BionetworkSBML(std::string name) :
    timeStepSize(-1),
    odeModel(0),
    settings(0) {

    modelKey = "";
    modelName = name;
    
    odeModel = new soslib_OdeModel();
    settings = new soslib_CvodeSettings();
}

BionetworkSBML::BionetworkSBML(std::string name, std::string _sbmlModelPath) :
    fileName(_sbmlModelPath),
    timeStepSize(-1),
    odeModel(0),
    settings(0) {
    
    modelKey = "";
    modelName = name;
    
    odeModel = new soslib_OdeModel(fileName);
    settings = new soslib_CvodeSettings();
}

BionetworkSBML::BionetworkSBML(std::string name, std::string _sbmlModelPath, double _timeStepSize) : 
    fileName(_sbmlModelPath),
    timeStepSize(_timeStepSize),
    odeModel(0),
    settings(0) {
    
    modelKey = "";
    modelName = name;
    
    odeModel = new soslib_OdeModel(fileName);
    
    if ( odeModel != 0 ){
        std::cout << "soslib_OdeModel instance successfully created..." << std::endl;
    } else {
        std::cout << "Null pointer to soslib_OdeModel..." << std::endl;
    }
    
    if( timeStepSize > 0.0 ){
        settings = new soslib_CvodeSettings(timeStepSize, 1);
    } else {
        settings = new soslib_CvodeSettings(1.0, 1);
    }
    
    if ( settings != 0 ){
        std::cout << "soslib_CvodeSettings instance successfully created..." << std::endl;
    } else {
        std::cout << "Null pointer to soslib_CvodeSettings..." << std::endl;
    }
}

BionetworkSBML::~BionetworkSBML(){}

bool BionetworkSBML::hasVariable(std::string varName) const {
    return odeModel->hasVariable(varName);
}

void BionetworkSBML::printSBMLModelInfo(){
    std::cout << "Information for SBML model " << modelName << std::endl;
    std::cout << "\tFile name: " << fileName << std::endl;
    
    std::stringstream ss;
    ss << "\tTime step size: " << timeStepSize << std::endl;
    std::cout << ss.str();
}




