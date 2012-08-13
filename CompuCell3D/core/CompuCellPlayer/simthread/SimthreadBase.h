#ifndef SIMTHREADBASE_H
#define SIMTHREADBASE_H

#include <string>
#include <GraphicsDataFields.h>

namespace CompuCell3D{
class Simulator;
class Dim3D;
};

class SimthreadBase{
    public:
		SimthreadBase(){};
		virtual ~SimthreadBase(){};
		virtual CompuCell3D::Simulator *getSimulator(){return 0;}
		virtual void setSimulator(CompuCell3D::Simulator * _simulator){}
		virtual std::string getSimulationFileName(){return "";};
		virtual std::string getSimulationPythonScriptName(){return "";};
		virtual void preStartInit(){};
		virtual void postStartInit(){};
                virtual void clearGraphicsFields(){}    
		virtual void loopWork(unsigned int _step){};
		virtual void loopWorkPostEvent(unsigned int _step){};

		virtual unsigned int getScreenUpdateFrequency(){return 0;}
		virtual GraphicsDataFields::floatField3D_t * createFloatFieldPy(CompuCell3D::Dim3D &_fieldDim,std::string _name){return 0;}
		virtual void fillPressureVolumeFlexPy(GraphicsDataFields::floatField3D_t & _floatField3D){}
		virtual GraphicsDataFields::vectorFieldCellLevel_t * createVectorFieldCellLevelPy(std::string _fieldName){return 0;}	
    
		virtual bool getStopSimulation(){return false;}
		virtual void setStopSimulation(bool *_stopSimulation){}
		virtual void sendStopSimulationRequest(){}

		virtual void handleErrorMessage(std::string _errorCategory, std::string  _error){};
	private:
		bool *pstopSimulation;
};

#endif

