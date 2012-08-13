#ifndef MAINCC3D_H
#define MAINCC3D_H
#include "transaction.h"
#include <string>
// #include <QtGui>
// #include <qpen.h>
// #include <qbrush.h>
#include <map>
#include <vector>

#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/EnergyFunction.h>
#include <CompuCell3D/PluginManager.h>
// #include <CompuCell3D/Plugin.h>

#include "GraphicsData.h"
#include "GraphicsDataFields.h"
#include "Graphics2D.h"

#include "PythonConfigureData.h"

#include "SimthreadBase.h"
#include "simthreadDLLSpecifier.h"

class QMutex;
class QSemaphore;

//template <typename Y> class CompuCell3D::Field3DImpl;
class CompuCell3D::Dim3D;
class CompuCell3D::EnergyFunction;
class CompuCell3D::Simulator;

namespace CompuCell3D{
   class Plugin;
};


template <typename Y>
class BasicArray;
class PyScriptRunner;

class SIMTHREAD_EXPORT  CC3DTransaction : public Transaction ,public SimthreadBase
{
public:
    CC3DTransaction(std::string  _filename);
    virtual ~CC3DTransaction();
//     QImage apply(const QImage &image);
    void applySimulation(bool *_stopSimulation);//TEMP
//	void applySimulation();
    QString messageStr(){return QString("DONE");};
    void setPauseMutexPtr(QMutex * mutexPtr){pauseMutexPtr=mutexPtr;};
    void setFieldDrawMutexPtr(QMutex * mutexPtr){fieldDrawMutexPtr=mutexPtr;};
    void setTransactionMutexPtr(QMutex * mutexPtr){mutexTransactionPtr=mutexPtr;};
    void setBufferFreeSem(QSemaphore * _bufferFillFreeSemPtr){bufferFillFreeSemPtr=_bufferFillFreeSemPtr;}
    void setBufferUsedSem(QSemaphore * _bufferFillUsedSemPtr){bufferFillUsedSemPtr=_bufferFillUsedSemPtr;}

    void setGraphicsDataFieldPtr(GraphicsDataFields * _graphFieldsPtr){graphFieldsPtr=_graphFieldsPtr;}

    void setScreenUpdateFrequency(unsigned int _freq){screenUpdateFrequency=_freq;}

	// From SimthreadBase
    virtual CompuCell3D::Simulator *getSimulator(){return simulator;}
    virtual void setSimulator(CompuCell3D::Simulator * _simulator){simulator=_simulator;}
    virtual void preStartInit();
    virtual void postStartInit();
    virtual void clearGraphicsFields();
    virtual void loopWork(unsigned int _step);
    virtual void loopWorkPostEvent(unsigned int _step);
    virtual std::string getSimulationFileName(){return filename;}
	 virtual std::string getSimulationPythonScriptName(){
		 if (runPythonFlag)
			return pyDataConf.pythonFileName.toStdString();
		 else
			 return "";
	 }

    virtual unsigned int getScreenUpdateFrequency(){return screenUpdateFrequency;}
    virtual GraphicsDataFields::floatField3D_t * createFloatFieldPy(CompuCell3D::Dim3D &_fieldDim,std::string _name);
    virtual GraphicsDataFields::vectorFieldCellLevel_t * createVectorFieldCellLevelPy(std::string _fieldName); 
	
	virtual bool getStopSimulation();
	virtual void setStopSimulation(bool *_stopSimulation){pstopSimulation = _stopSimulation;}
	virtual void sendStopSimulationRequest();

	virtual void handleErrorMessage(std::string _errorCategory, std::string  _error);

    virtual void fillPressureVolumeFlexPy(GraphicsDataFields::floatField3D_t & _floatField3D);
      
    void registerPyScriptRunner(PyScriptRunner * _pyScriptRunner){pyScriptRunner=_pyScriptRunner;}

    void simulationThreadCpp();
    void simulationThreadPython();

    void setRunPythonFlag(bool _flag){runPythonFlag=_flag;}    
    bool getRunPythonFlag(){return runPythonFlag;}

    void setUseXMLFileFlag(bool _flag){useXMLFileFlag=_flag;}
    bool getUseXMLFileFlag(){return useXMLFileFlag;}

    void setPyDataConf(PythonConfigureData & _pyDataConf){pyDataConf=_pyDataConf;}
    PythonConfigureData getPyDataConf(){return pyDataConf;}


private:
    std::string filename;
	 std::string pythonScriptNameFromPlayer;

    PythonConfigureData pyDataConf;
    PyScriptRunner *pyScriptRunner;
    

    unsigned int zPosition;
    unsigned int screenUpdateFrequency;
    
    GraphicsDataFields * graphFieldsPtr;

    void fillField3D(GraphicsDataFields  &graphFields);
    void createConcentrationFields(GraphicsDataFields  &graphFields, std::map<std::string,
         CompuCell3D::Field3DImpl<float>*> & _fieldMap);
   void fillConcentrationFields(GraphicsDataFields  &graphFields, std::map<std::string,CompuCell3D::Field3DImpl<float>*> & _fieldMap);
   
   void createPreasureFields(  GraphicsDataFields  &graphFields,
                                             CompuCell3D::Dim3D fieldDim,
                                             BasicArray<CompuCell3D::EnergyFunction *>  & energyFunctions
                                             );
   void fillPressureFields(GraphicsDataFields  &graphFields);
   
   void fillPressureVolume( GraphicsDataFields  &graphFields,
                                             GraphicsDataFields::floatField3D_t & _floatField3D,
                                             CompuCell3D::EnergyFunction * energyFunctionPtr);
                                             
   void fillPressureVolumeFlex( GraphicsDataFields  &graphFields,
                                             GraphicsDataFields::floatField3D_t & _floatField3D,
                                             CompuCell3D::EnergyFunction * energyFunctionPtr);
   
   void createVectorCellFields(  GraphicsDataFields  &graphFields,
                                             CompuCell3D::Dim3D fieldDim,
                                             BasicArray<CompuCell3D::EnergyFunction *>  & energyFunctions,
                                             CompuCell3D::PluginManager<CompuCell3D::Plugin>::plugins_t & pluginMap
                                             );
                                             
    void fillVectorCellFields(GraphicsDataFields  &graphFields);
    void fillVelocity( GraphicsDataFields  &graphFields,
                                             GraphicsDataFields::vectorCellFloatField3D_t & _vectorCellFloatField3D,
                                             CompuCell3D::Plugin * plugin
                                             );
                      
    void markBorder(const CompuCell3D::Point3D & _pt , CompuCell3D::CellG *_currentCell, GraphicsData * _pixel);

    
    CompuCell3D::Field3D<CompuCell3D::CellG *> *cellFieldG;
    
    // volume plugin names - we derive pressure using volume plugin data
    std::map<std::string, CompuCell3D::EnergyFunction *> pressureNameEnergyMap;
    std::map<std::string, CompuCell3D::Plugin *> pluginNameMap;
    

    QMutex * pauseMutexPtr;
    QMutex * fieldDrawMutexPtr;
    QMutex * mutexTransactionPtr;
    QSemaphore * bufferFillFreeSemPtr;
    QSemaphore * bufferFillUsedSemPtr;

	// Flag that stop simulation
	bool *pstopSimulation;
	

    CompuCell3D::Simulator *simulator;
    bool runPythonFlag;
    bool useXMLFileFlag;
};


#endif
