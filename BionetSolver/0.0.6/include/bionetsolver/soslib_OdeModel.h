
#ifndef SOSLIB_ODEMODEL_H
#define SOSLIB_ODEMODEL_H

#include "sbmlsolver/odeModel.h"
#include "sbmlsolver/odeConstruct.h"
#include "soslib_OdeModel.h"
#include "BionetworkDLLSpecifier.h"

   class BIONETWORK_EXPORT soslib_OdeModel{
        private:
            odeModel_t* om;
            Model_t* getModel() const ;
        public:
            soslib_OdeModel();
            ~soslib_OdeModel();
            soslib_OdeModel(std::string);
            void createOdeModel(const char *);
            odeModel_t* getOdeModel() const ;
            void printParameterValues() const ;
            int numAllVariables() const ;
            int numAssignments() const ;
            int numConstants() const ;
            int numStateVariables() const ;
            vector<variableIndex_t *> getStateVariableIndexes() const ;
            vector<variableIndex_t *> getStateVariableIndexes(vector<std::string> var_names) const ;
            
            vector<variableIndex_t *> getParameterVariableIndexes() const ;
            vector<variableIndex_t *> getParameterVariableIndexes(vector<std::string> param_names) const ;
            
            variableIndex_t * getVariableIndex(std::string) const ;
            
            std::string getStateVariablesAsString() const ;
            std::string getParametersAsString() const ;
            
            bool hasVariable(std::string) const ;
    };

#endif



