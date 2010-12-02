
#ifndef BIONETWORK_H
#define BIONETWORK_H

#include <map>
#include "bionetsolver/BionetworkTemplateLibrary.h"
#include "BionetworkDLLSpecifier.h"

class soslib_IntegratorInstance;
class BionetworkUtilManager;

class BIONETWORK_EXPORT Bionetwork{
    private:
        //std::pair<std::string, const BionetworkTemplateLibrary *> cellType;
        std::pair<std::string, const BionetworkTemplateLibrary *> templateLibrary;
        std::map<std::string, soslib_IntegratorInstance *> integrators;
        
        //double divideVolume;
        //double initialVolume;
        
        //std::string cellTypeAsString() const ;
        std::string templateLibraryAsString() const ;
        
        const Bionetwork * getConstPointer() const ;
        
        //void setCellType(const BionetworkTemplateLibrary *);
        void setTemplateLibrary(const BionetworkTemplateLibrary *);
        
        void addNewIntegrator(const BionetworkSBML *);
        
        BionetworkUtilManager* utilManager;
        
    public:
        Bionetwork();
        Bionetwork(std::pair<std::string, const BionetworkTemplateLibrary *> ) ;
        Bionetwork(std::string, const BionetworkTemplateLibrary *) ;
        Bionetwork( Bionetwork * );
        ~Bionetwork() ;
        
        //void updateIntracellularState();
        //void updateIntracellularStateWithTimeStep( double );
        void updateBionetworkState();
        void updateBionetworkStateWithTimeStep( double );
        
        void initializeIntegrators();
        void initializeIntegrators(std::map<std::string, const BionetworkSBML *>);
        //void setIntracellularState(std::map<std::string, double>);
        void setBionetworkState(std::map<std::string, double>);
        
        //void setCellPropertyValues( std::map<std::string, double> );
        //void setCellPropertyValue(std::pair<std::string, double> );
        //double getCellPropertyAsDouble(std::string ) const ;
        //std::pair<std::string, double> getCellPropertyAsPair(std::string ) const ;
        
        void setPropertyValue( std::string, double );
        std::pair<bool, double> findPropertyValue( std::string ) const ;
        //std::pair<bool, double> findCellPropertyValue( std::string ) const ;
        
        //void changeCellType(const BionetworkTemplateLibrary * );
        void changeTemplateLibrary(const BionetworkTemplateLibrary * );
        
        //double getDivideVolume() const { return divideVolume; };
        //void setDivideVolume(double vol){ divideVolume = vol; };
        //double getInitialVolume() const { return initialVolume; };
        //void setInitialVolume(double vol){ initialVolume = vol; };
        
        //std::string getCellTypeName() const ;
        std::string getTemplateLibraryName() const ;
        
        //const BionetworkTemplateLibrary * getCellTypeInstancePtr() const { return cellType.second; };
        const BionetworkTemplateLibrary * getTemplateLibraryInstancePtr() const { return templateLibrary.second; };
        
        //void printIntracellularState() const ;
        //void printIntracellularState(bool) const ;
        void printBionetworkState() const ;
        void printBionetworkState(bool) const ;
        
        //std::string getIntracellularStateAsString( bool ) const ;
        //std::string getIntracellStateVarNamesAsString( std::string ) const ;
        //std::string getIntracellStateAsString( std::string ) const ;
        std::string getBionetworkStateAsString( bool ) const ;
        std::string getBionetworkStateVarNamesAsString( std::string ) const ;
        std::string getBionetworkStateAsString( std::string ) const ;
        
        bool hasSBMLModelByKey(std::string) const ;
        std::vector<std::string> getSBMLModelNames() const;
        
        const soslib_IntegratorInstance* getIntegrInstance(std::string) const ;
        std::map<std::string, const soslib_IntegratorInstance *> getIntegrInstances() const ;
        
        void printMessage() const ;
};


#endif




