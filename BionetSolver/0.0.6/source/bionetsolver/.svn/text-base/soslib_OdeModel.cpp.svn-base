
#include <iostream>

#include "bionetsolver/soslib_OdeModel.h"

    soslib_OdeModel::soslib_OdeModel(){
        om = NULL;
        //std::cout << "New OdeModel object created." << std::endl;
    }
    
    soslib_OdeModel::~soslib_OdeModel(){
        //std::cout << "Called soslib_OdeModel destructor..." << std::endl;
        if( om != NULL){
            //std::cout << "Calling ODEModel_free..." << std::endl;
            ODEModel_free(om);
        }
    }
    
    soslib_OdeModel::soslib_OdeModel(std::string filePath){
        om = NULL;
        
        //std::cout << "Calling ODEModel_createFromFile with file name ";
        std::cout << filePath.c_str() << "..." << std::endl;
        
        om = ODEModel_createFromFile(filePath.c_str());
        
        if (om == NULL) std::cout << "odeModel creation unsuccessful. Null pointer for odeModel" << std::endl;
        else std::cout << "odeModel creation successful." << std::endl;
    }
    
    void soslib_OdeModel::createOdeModel(const char * filePath){
        if (om != NULL){
            std::cout << "om is *not* NULL" << std::endl;
            ODEModel_free(om);
        }
        om = ODEModel_createFromFile(filePath);
        
        if (om != NULL) ODEModel_dumpNames(om);
        else std::cout << "Null pointer for odeModel" << std::endl;
        
        //std::cout << "createOdeModel was called." << std::endl;
    }
    
    odeModel_t* soslib_OdeModel::getOdeModel() const {
        if (om == NULL) 
            std::cout << "OdeModel has not been initialized." << std::endl;
        return om;
    }
    
    Model_t* soslib_OdeModel::getModel() const {
        Model_t* model = NULL;
        if (om == NULL){
            std::cout << "OdeModel has not been initialized." << std::endl;
        } else {
            model = om->m;
        }
        return model;
    }
    
    int soslib_OdeModel::numAllVariables() const {
        int numAllVar;
        if (om != 0) numAllVar = ODEModel_getNumValues(om);
        else{
            numAllVar = 0;
            std::cout << "Null pointer for odeModel" << std::endl;
        }
        return numAllVar;
    }
    
    int soslib_OdeModel::numAssignments() const {
        int numAssn;
        if (om != 0) numAssn = ODEModel_getNumAssignments(om);
        else{
            numAssn = 0;
            std::cout << "Null pointer for odeModel" << std::endl;
        }
        return numAssn;
    }
    
    int soslib_OdeModel::numConstants() const {
        int numConst;
        if (om != 0) numConst = ODEModel_getNumConstants(om);
        else{
            numConst = 0;
            std::cout << "Null pointer for odeModel" << std::endl;
        }
        return numConst;
    }

void soslib_OdeModel::printParameterValues() const {
    std::cout << "New printParameterValues function called:" << std::endl;
    if (om != NULL){
        vector<variableIndex_t *> paramIndexes(this->getParameterVariableIndexes());
        for (unsigned int i = 0; i < paramIndexes.size(); i++){
            std::cout << VariableIndex_getName(paramIndexes.at(i), om) << ":\t\t";
            std::cout << 
                Model_getValueById(this->getModel(),
                    VariableIndex_getName(paramIndexes.at(i), om)) << std::endl;
        }
        std::cout << std::endl;
    }else std::cout << "OdeModel object not created yet." << std::endl << std::endl;
}

int soslib_OdeModel::numStateVariables() const {
    return (ODEModel_getNumValues(om)
        - ODEModel_getNumConstants(om)
        - ODEModel_getNumAssignments(om));
}

vector<variableIndex_t *> soslib_OdeModel::getStateVariableIndexes() const {
    vector<variableIndex_t *> varIndexes;
    if (om != NULL)
        for (int i=0; i < this->numStateVariables(); i++)
            varIndexes.push_back(ODEModel_getOdeVariableIndex(om, i));
    return varIndexes;
}

vector<variableIndex_t *> soslib_OdeModel::getStateVariableIndexes(vector<std::string> var_names) const {
    vector<variableIndex_t *> varIndexes;
    if (om != NULL)
        for (unsigned int i=0; i < var_names.size(); i++)
            varIndexes.push_back(ODEModel_getVariableIndex(om, var_names[i].c_str()));
    return varIndexes;
}

vector<variableIndex_t *> soslib_OdeModel::getParameterVariableIndexes() const {
    vector<variableIndex_t *> paramIndexes;
    if (om != NULL)
        for (int i=0; i < this->numConstants(); i++)
            paramIndexes.push_back(ODEModel_getConstantIndex(om, i));
    return paramIndexes;
}

vector<variableIndex_t *> soslib_OdeModel::getParameterVariableIndexes(vector<std::string> param_names) const {
    vector<variableIndex_t *> paramIndexes;
    if (om != NULL)
        for (unsigned int i=0; i < param_names.size(); i++)
            paramIndexes.push_back(ODEModel_getVariableIndex(om, param_names[i].c_str()));
    return paramIndexes;
}

variableIndex_t * soslib_OdeModel::getVariableIndex(std::string varName) const {
    if (om != NULL)
        return ODEModel_getVariableIndex(om, varName.c_str());
    else
        return NULL;
}

std::string soslib_OdeModel::getStateVariablesAsString() const {
    std::string stateVariableString("");
    if (om != NULL){
        vector<variableIndex_t *> varIndexes(this->getStateVariableIndexes());
        stateVariableString += VariableIndex_getName(varIndexes.at(0), om);
        
        for (unsigned int i = 1; i < varIndexes.size(); i++){
            stateVariableString += "\t";
            stateVariableString += VariableIndex_getName(varIndexes.at(i), om);;
        }
    }
    
    std::cout << "New getStateVariableAsString function called. Returning string: " << std::endl;
    std::cout << stateVariableString << std::endl;
    return stateVariableString;
}

std::string soslib_OdeModel::getParametersAsString() const {
    std::string parameterString("");
    if (om != NULL){
        vector<variableIndex_t *> varIndexes(this->getParameterVariableIndexes());
        parameterString += VariableIndex_getName(varIndexes.at(0), om);
        
        for (unsigned int i = 1; i < varIndexes.size(); i++){
            parameterString += "\t";
            parameterString += VariableIndex_getName(varIndexes.at(i), om);;
        }
    }
    return parameterString;
}

bool soslib_OdeModel::hasVariable(std::string varName) const {
    bool variableFound = false;
    if (om != NULL){
        vector<variableIndex_t *>::iterator itr;
        
        vector<variableIndex_t *> varIndexes( getStateVariableIndexes() );
        itr = varIndexes.begin();
        for(; itr != varIndexes.end(); ++itr){
            if ( varName == VariableIndex_getName(*itr, om) ){
                variableFound = true;
                break;
            }
        }
        
        if( !variableFound ){
            varIndexes = getParameterVariableIndexes();
            itr = varIndexes.begin();
            for(; itr != varIndexes.end(); ++itr){
                if ( varName == VariableIndex_getName(*itr, om) ){
                    variableFound = true;
                    break;
                }
            }
        }
    }
    return variableFound;
}
