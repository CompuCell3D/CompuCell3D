
#ifndef SOSLIB_INTEGRATORINSTANCE_H
#define SOSLIB_INTEGRATORINSTANCE_H

#include "sbmlsolver/integratorInstance.h"
#include "soslib_OdeModel.h"
#include "soslib_CvodeSettings.h"
#include "BionetworkDLLSpecifier.h"

   class BionetworkUtilManager;
   
   class BIONETWORK_EXPORT soslib_IntegratorInstance{
    private:
        integratorInstance_t* ii;
        const soslib_OdeModel* odeModel;
        soslib_CvodeSettings* settings;
        std::string modelKey;
        std::string modelName;
        BionetworkUtilManager* utilManager;
        
    public:
        soslib_IntegratorInstance();
        soslib_IntegratorInstance(const soslib_OdeModel *, const soslib_CvodeSettings *);
        ~soslib_IntegratorInstance();
        
        void createIntegratorInstance(const soslib_OdeModel *, const soslib_CvodeSettings *);
        
        void setModelKey(std::string key) { modelKey = key; } ;
        std::string getModelKey() const { return modelKey; };
        
        void setModelName(std::string name) { modelName = name; };
        std::string getModelName() const { return modelName; };
        
        void setStateValue( std::string, double );
        void setStateValue( std::pair<std::string, double> );
        
        std::map<std::string, double> getState() const ;
        void setState(std::map<std::string, double> );
        void setStateDirect(const std::map<std::string, double> &);
        std::map<std::string, double> getParamValues() const ;
        void setParamValues(std::map<std::string, double> );
        void setParamsDirect(const std::map<std::string, double> &);
            
        double getValueAsDouble( std::string ) const ;
        std::pair<bool, double> findValueAsDouble( std::string ) const ;
        
        //std::string getOdeModelKey() const { return odeModel->getModelKey(); };
        
        std::string getStateAsString();
        pair<std::string, std::string> getStateAsString(bool);
        pair<std::string, std::string> getParamValuesAsString(bool);
        void printIntegrationResults();
        
        //bool validCurrentEndTime() const ;
        
        bool indefiniteIntegrationIsSet() const ;
        void setIndefiniteIntegration(int);
        
        void setNextTimeStep( double );
        double getNextTimeStep() const ;
        void setCurrentEndTime(double);
        double getCurrentEndTime() const ;
        
        void integrateOneStep();
        void resetIntegrator();
        void setIntegrator( const soslib_CvodeSettings * );
        void integrateOneStepAndResetIntegrator();
        
        integratorInstance_t* getAddressOfIntegratorInstance() { return ii; };
        
        const soslib_OdeModel* getOdeModel() const { return odeModel; };
        const soslib_CvodeSettings* getSettings() const { return settings; };
   };

#endif



