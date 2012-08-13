#include <iostream>
#include <typeinfo>

#include <Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicSmartPointer.h>

#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
//#include <CompuCell3D/plugins/Volume/VolumeEnergy.h>
// #include <CompuCell3D/plugins/VolumeFlex/VolumeFlexEnergy.h>
 #include <CompuCell3D/plugins/CellVelocity/CellVelocityData.h>
 #include <CompuCell3D/plugins/CellVelocity/CellVelocityPlugin.h>
#include <CompuCell3D/plugins/PlayerSettings/PlayerSettingsPlugin.h>
#include <CompuCell3D/plugins/PlayerSettings/PlayerSettings.h>


#include <BasicUtils/BasicClassAccessor.h>

//#include <XMLCereal/XMLPullParser.h>

// #include <XercesUtils/XercesStr.h>
// #include <xercesc/util/PlatformUtils.hpp>
// XERCES_CPP_NAMESPACE_USE;

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
using namespace std;

#include <stdlib.h>
 #include <PyScriptRunner.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include "simthreadAccessor.h"


#include "mainCC3D.h"

SimthreadBase* simthreadBasePtr;
double numberGlobal;

//extern SimthreadBase* simthreadBasePtr;

void Syntax(const string name) {
  cerr << "Syntax: " << name << " <config>" << endl;
  exit(1);
}

//using namespace std;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC3DTransaction::CC3DTransaction(std::string  _filename)
{
	///Filename
	screenUpdateFrequency=1;
	filename=_filename;
	runPythonFlag=false;
	useXMLFileFlag=false;
	simthreadBasePtr=this; // Very important!

	pyScriptRunner=0;

	cerr<<"This is the file name:"<<filename<<endl;

   
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC3DTransaction::~CC3DTransaction(){
	cerr<<"\n\n\n\n \t\t\tCALLING DESTRUCTOR FOR CC3DTransaction"<<endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void  CC3DTransaction::createConcentrationFields(GraphicsDataFields  &graphFields ,
      std::map<std::string,CompuCell3D::Field3DImpl<float>*> & _fieldMap)
{

   std::map<std::string,Field3DImpl<float>*>::iterator mitr;
   
   
   for( mitr = _fieldMap.begin(); mitr != _fieldMap.end() ; ++mitr){
      cerr<<"GOT FIELD: "<<mitr->first<<endl;

      CompuCell3D::Dim3D fieldDim=mitr->second->getDim();
      cerr<<"Dimension:"<<fieldDim<<endl;
      
      graphFields.allocateFloatField3D(fieldDim.x, fieldDim.y, fieldDim.z , mitr->first); //here I alocate concentration field that player will access
      graphFields.insertPlotNamePlotTypePair(mitr->first,"scalar");//registering plot name - plot type pair
   }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool CC3DTransaction::getStopSimulation(){
	return *pstopSimulation;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CC3DTransaction::clearGraphicsFields(){
   graphFieldsPtr->clearAllocatedFields();
}


void CC3DTransaction::fillConcentrationFields(GraphicsDataFields  &graphFields, std::map<std::string,CompuCell3D::Field3DImpl<float>*> & _fieldMap){

   ///set mutextes
   fieldDrawMutexPtr->lock();


   std::map<std::string,GraphicsDataFields::floatField3D_t*> & floatField3DNameMap=graphFields.getFloatField3DNameMap();
   
   std::map<std::string,GraphicsDataFields::floatField3D_t*>::iterator floatFieldMitr;
   std::map<std::string,CompuCell3D::Field3DImpl<float>*>::iterator mitr;
   
   for( mitr = _fieldMap.begin(); mitr != _fieldMap.end() ; ++mitr){
      floatFieldMitr = floatField3DNameMap.find(mitr->first);
      
      if(floatFieldMitr == floatField3DNameMap.end()){
         cerr<<"Could not find element "<<mitr->first<<" in the floatField3DNameMap."<<endl;
         continue;
      }

      GraphicsDataFields::floatField3D_t & floatField3D = *floatFieldMitr->second;
      CompuCell3D::Field3DImpl<float> & concentrationField = *mitr->second;

      

      CompuCell3D::Point3D pt;
      CompuCell3D::Point3D fieldDim=mitr->second->getDim();
      
      for(unsigned int x = 0 ; x < fieldDim.x ; ++x)
         for(unsigned int y = 0 ; y < fieldDim.y ; ++y)
            for(unsigned int z = 0 ; z < fieldDim.z ; ++z){
            pt.x=x;
            pt.y=y;
            pt.z=z;      
            
            floatField3D[x][y][z]=concentrationField.get(pt);
                      
         }
   }

    ///unset
   fieldDrawMutexPtr->unlock();

   
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GraphicsDataFields::floatField3D_t * CC3DTransaction::createFloatFieldPy(CompuCell3D::Dim3D& _fieldDim,std::string _fieldName){
 
   std::map<std::string,GraphicsDataFields::floatField3D_t*> & floatField3DNameMap=graphFieldsPtr->getFloatField3DNameMap();

   graphFieldsPtr->allocateFloatField3D(_fieldDim.x, _fieldDim.y, _fieldDim.z , _fieldName);
   graphFieldsPtr->insertPlotNamePlotTypePair(_fieldName,"scalar");//registering plot name - plot type pair

   std::map<std::string,GraphicsDataFields::floatField3D_t*>::iterator floatFieldMitr;

   floatFieldMitr = floatField3DNameMap.find(_fieldName);//getting field ptr for appropriate pressure field
   if(floatFieldMitr == floatField3DNameMap.end()){
      cerr<<"Could not find element "<<_fieldName<<" in the floatField3DNameMap."<<endl;
      
   }
   return floatFieldMitr->second;
   


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GraphicsDataFields::vectorFieldCellLevel_t * CC3DTransaction::createVectorFieldCellLevelPy(std::string _fieldName){
   
   std::map<std::string,GraphicsDataFields::vectorFieldCellLevel_t *> & vectorFieldCellLevelMap=graphFieldsPtr->getVectorFieldCellLevelNameMap();
   
   graphFieldsPtr->allocateVectorFieldCellLevel(_fieldName);
   graphFieldsPtr->insertPlotNamePlotTypePair(_fieldName,"vector_cell_level");//registering plot name - plot type pair

   std::map<std::string,GraphicsDataFields::vectorFieldCellLevel_t *>::iterator vitr;
   vitr=vectorFieldCellLevelMap.find(_fieldName);
   if( vitr == vectorFieldCellLevelMap.end() ){
      cerr<<"Could not find element "<<_fieldName<<" in the vectorFieldCellLevelMap"<<endl;
   }

   return vitr->second;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::fillPressureVolumeFlexPy(GraphicsDataFields::floatField3D_t & _floatField3D){

Point3D pt;
CompuCell3D::CellG * cell=0;

   for(unsigned int x = 0 ; x < graphFieldsPtr->getSizeL() ; ++x)
      for(unsigned int y = 0 ; y < graphFieldsPtr->getSizeM() ; ++y)
         for(unsigned int z = 0 ; z < graphFieldsPtr->getSizeN() ; ++z){
         pt.x=x;
         pt.y=y;
         pt.z=z;
         
         cell=cellFieldG->get(pt);

         if(!cell){
            _floatField3D[x][y][z]=0.0;
            continue;
         }

         _floatField3D[x][y][z]=cell->targetVolume - cell->volume;
          
      }
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::createPreasureFields(  GraphicsDataFields  &graphFields,
                                             CompuCell3D::Dim3D fieldDim,
                                             BasicArray<CompuCell3D::EnergyFunction *>  & energyFunctions)
{
   string energyFunctionName;
   string::size_type N_POS=string::npos;
   
   for(int i  = 0 ; i < energyFunctions.getSize() ; ++i ){

      energyFunctionName=energyFunctions[i]->toString();

      if(energyFunctionName.find("Volume")!=N_POS){
      
         pressureNameEnergyMap.insert(make_pair(energyFunctionName,energyFunctions[i]));
         string fieldName("Pressure_");
         fieldName+=energyFunctionName;
         graphFields.allocateFloatField3D(fieldDim.x, fieldDim.y, fieldDim.z , fieldName);
         graphFields.insertPlotNamePlotTypePair(fieldName,"scalar");//registering plot name - plot type pair
      }
      
   }


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::fillPressureFields(GraphicsDataFields  &graphFields){

   ///set mutextes
   fieldDrawMutexPtr->lock();


   std::map<std::string,GraphicsDataFields::floatField3D_t*> & floatField3DNameMap=graphFields.getFloatField3DNameMap();

   std::map<std::string,GraphicsDataFields::floatField3D_t*>::iterator floatFieldMitr;
   map<string, CompuCell3D::EnergyFunction *>::iterator mitr;

   for( mitr = pressureNameEnergyMap.begin(); mitr != pressureNameEnergyMap.end() ; ++mitr){

      string fieldName("Pressure_");
      fieldName+=mitr->first;

      floatFieldMitr = floatField3DNameMap.find(fieldName);//getting field ptr for appropriate pressure field
      if(floatFieldMitr == floatField3DNameMap.end()){
         cerr<<"Could not find element "<<mitr->first<<" in the floatField3DNameMap."<<endl;
         continue;
      }

      GraphicsDataFields::floatField3D_t & floatField3D = *floatFieldMitr->second;


      if(mitr->first=="Volume"){

         fillPressureVolume(graphFields , floatField3D , mitr->second);

      }
      else if(mitr->first=="VolumeFlex"){
         fillPressureVolumeFlex(graphFields , floatField3D , mitr->second);
      }
   }

    ///unset
   fieldDrawMutexPtr->unlock();

   
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::fillPressureVolume( GraphicsDataFields  &graphFields,
                                          GraphicsDataFields::floatField3D_t & _floatField3D,
                                          CompuCell3D::EnergyFunction * energyFunctionPtr                                          
){

// VolumeEnergy * volEnergy = (VolumeEnergy*)energyFunctionPtr;
// 
// double targetVolume = volEnergy->getTargetVolume();
// 
// Point3D pt;
// CompuCell3D::CellG * cell=0;
// 
//    for(unsigned int x = 0 ; x < graphFields.getSizeL() ; ++x)
//       for(unsigned int y = 0 ; y < graphFields.getSizeM() ; ++y)
//          for(unsigned int z = 0 ; z < graphFields.getSizeN() ; ++z){
//          pt.x=x;
//          pt.y=y;
//          pt.z=z;
//          
//          cell=cellFieldG->get(pt);
// 
//          if(!cell){
//             _floatField3D[x][y][z]=0.0;
//             continue;
//          }
// 
//          
//          
//          _floatField3D[x][y][z]=targetVolume - cell->volume;
//       }


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::fillPressureVolumeFlex( GraphicsDataFields  &graphFields,
                                          GraphicsDataFields::floatField3D_t & _floatField3D,
                                          CompuCell3D::EnergyFunction * energyFunctionPtr                                          
){

// VolumeFlexEnergy * volFlexEnergy = (VolumeFlexEnergy*)energyFunctionPtr;
// 
// const std::vector<VolumeEnergyParam> & volumeEnergyParamVector= volFlexEnergy->getVolumeEnergyParamVector();
// 
// 
// Point3D pt;
// CompuCell3D::CellG * cell=0;
// 
//    for(unsigned int x = 0 ; x < graphFields.getSizeL() ; ++x)
//       for(unsigned int y = 0 ; y < graphFields.getSizeM() ; ++y)
//          for(unsigned int z = 0 ; z < graphFields.getSizeN() ; ++z){
//          pt.x=x;
//          pt.y=y;
//          pt.z=z;
//          
//          cell=cellFieldG->get(pt);
// 
//          if(!cell){
//             continue;
//          }
// 
//          _floatField3D[x][y][z]=volumeEnergyParamVector[cell->type].targetVolume - cell->volume;
//           
//       }


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void CC3DTransaction::createVectorCellFields(  GraphicsDataFields  &graphFields,
                                             CompuCell3D::Dim3D fieldDim,
                                             BasicArray<CompuCell3D::EnergyFunction *>  & energyFunctions,
                                             CompuCell3D::PluginManager<Plugin>::plugins_t & pluginMap
)
{
   
   string fieldName;
      
      for(PluginManager<Plugin>::plugins_t::iterator/*pluginMapItr_t*/ itr = pluginMap.begin() ; itr != pluginMap.end() ; ++itr){
         if(itr->second && itr->second->toString()=="CellVelocity"){

            fieldName="CellVelocity";
            graphFields.allocateVectorCellFloatField3D(fieldDim.x, fieldDim.y, fieldDim.z , fieldName);
            graphFields.insertPlotNamePlotTypePair(fieldName,"vector_cell_level");//registering plot name - plot type pair
            pluginNameMap.insert(make_pair(fieldName,itr->second));

         }
      
      }
      
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CC3DTransaction::fillVectorCellFields(GraphicsDataFields  &graphFields){
   ///set mutextes
   fieldDrawMutexPtr->lock();
//    cerr<<"MAIN: GOT fieldDrawMutexPtr "<<endl;

   std::map<std::string,GraphicsDataFields::vectorCellFloatField3D_t*> &
      vectorCellFloatField3DNameMap=graphFields.getVectorCellFloatField3DNameMap();

   
   std::map<std::string,GraphicsDataFields::vectorCellFloatField3D_t*>::iterator mitr;
   string fieldName;
   
   for( mitr = vectorCellFloatField3DNameMap.begin(); mitr != vectorCellFloatField3DNameMap.end() ; ++mitr){

      
      fieldName=mitr->first;
      GraphicsDataFields::vectorCellFloatField3D_t & vectorCellFloatField3D = *mitr->second;
      


      if(fieldName=="CellVelocity"){
         std::map<std::string, CompuCell3D::Plugin *>::iterator pitr = pluginNameMap.find(fieldName);
         if(pitr != pluginNameMap.end() ){
            fillVelocity(graphFields , vectorCellFloatField3D , pitr->second);
         }else{
            cerr<<"Could not find pointer to "<<fieldName<<" related plugin"<<endl;
         }
      }
   }
         
   
   

    ///unset
   fieldDrawMutexPtr->unlock();
//    cerr<<"MAIN: RELEASE fieldDrawMutexPtr "<<endl;


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::fillVelocity( GraphicsDataFields  &graphFields,
                                          GraphicsDataFields::vectorCellFloatField3D_t & _vectorCellFloatField3D,
                                          CompuCell3D::Plugin * plugin)
{
	Point3D pt;
	CompuCell3D::CellG * cell=0;

	CompuCell3D::CellVelocityPlugin * cellVelocityPlugin;
	if(plugin->toString() == "CellVelocity"){
		cellVelocityPlugin = (CompuCell3D::CellVelocityPlugin*)(plugin);
	}else{
		cerr<<"Wrong Plugin - "<<plugin->toString()<<endl;
		exit(0);
	}



	BasicClassAccessor<CellVelocityData> * accessorPtr= cellVelocityPlugin -> getCellVelocityDataAccessorPtr();


	for(unsigned int x = 0 ; x < graphFields.getSizeL() ; ++x)
		for(unsigned int y = 0 ; y < graphFields.getSizeM() ; ++y)
			for(unsigned int z = 0 ; z < graphFields.getSizeN() ; ++z)
			{
				pt.x=x;
				pt.y=y;
				pt.z=z;

				cell=cellFieldG->get(pt);

				if(!cell){
					_vectorCellFloatField3D[x][y][z]=make_pair(Coordinates3D<float>(0.,0.,0.),cell);
					continue;
				}

			//_vectorCellFloatField3D[x][y][z]=make_pair(accessorPtr->get(cell->extraAttribPtr)->getInstantenousVelocity(),cell);
			//          cerr<<"vel="<<accessorPtr->get(cell->extraAttribPtr)->getAverageVelocity().X()<<endl;
			_vectorCellFloatField3D[x][y][z]=make_pair(accessorPtr->get(cell->extraAttribPtr)->getAverageVelocity(),cell);
			}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void CC3DTransaction::fillField3D(GraphicsDataFields  &graphFields){

Point3D pt;
CompuCell3D::CellG * cell=0;
   ///set mutextes
   fieldDrawMutexPtr->lock();
   
   for(unsigned int x = 0 ; x < graphFields.getSizeL() ; ++x)
      for(unsigned int y = 0 ; y < graphFields.getSizeM() ; ++y)
         for(unsigned int z = 0 ; z < graphFields.getSizeN() ; ++z){
         pt.x=x;
         pt.y=y;
         pt.z=z;
         
         cell=cellFieldG->get(pt);

         if(!cell){
            graphFields.field3DGraphicsData[x][y][z].type = 0;
            graphFields.field3DGraphicsData[x][y][z].id = 0;
            graphFields.field3DGraphicsData[x][y][z].flag = 0;
            graphFields.field3DGraphicsData[x][y][z].averageConcentration = 0;
            continue;
         }

         
         graphFields.field3DGraphicsData[x][y][z].type=cell->type;
         graphFields.field3DGraphicsData[x][y][z].id = (long)cell->id;
         graphFields.field3DGraphicsData[x][y][z].flag = cell->flag;
         graphFields.field3DGraphicsData[x][y][z].averageConcentration = cell->averageConcentration;
                   
      }
   ///unset   
   fieldDrawMutexPtr->unlock();
   
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::markBorder(const CompuCell3D::Point3D & _pt , CellG *_currentCell, GraphicsData * _pixel){

   Point3D pt;
   CellG *cell;
   vector<CompuCell3D::Point3D> offset(4);
   
   offset[0]=Point3D(0,0,-1);
   offset[1]=Point3D(0,0,1);
   offset[2]=Point3D(-1,0,0);
   offset[3]=Point3D(1,0,0);


   
   for( unsigned int i = 0 ; i < offset.size() ; ++i){
      pt=_pt;
      pt+=offset[i];
      
      cell=cellFieldG->get(pt);
      if(!cell)

      if(cell != _currentCell){
         _pixel->type  = 4;
         return;
      }
   }
   

   
   

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::preStartInit(){
   cerr<<"PRESTART EVENT"<<endl;
   Dim3D fieldDim=simulator->getPotts()->getCellFieldG()->getDim();

   
    ///initialization of the drawing utils
   //Freeing all previously used fields
//    graphFieldsPtr->~GraphicsDataFields();      
   
/*   graphFieldsPtr->clearFloatFields3D();
   graphFieldsPtr->clearVectorFieldsCellLevel();*/
   ///allocate fields 



   graphFieldsPtr->allocateField3D(fieldDim.x,fieldDim.y,fieldDim.z);//crating Cell Field
   graphFieldsPtr->insertPlotNamePlotTypePair("Cell Field","cell_field");
   createConcentrationFields(*graphFieldsPtr ,simulator->getConcentrationFieldNameMap());
//    createPreasureFields( *graphFieldsPtr , fieldDim , simulator->getPotts()->getEnergyFunctionsArray());
   
//    createVectorCellFields( *graphFieldsPtr , fieldDim , simulator->getPotts()->getEnergyFunctionsArray(), simulator->getPluginMap());
                                           
   
   ///assign field ptr
   cellFieldG=simulator->getPotts()->getCellFieldG();

   
   ///3 rd coordinate value
   zPosition=50;
   
   /// Now will set up drawing canvas sizes and configure the player 
   TransactionStartEvent *eventStart=new TransactionStartEvent();
   CompuCell3D::PlayerSettingsPlugin *playerSettingsPlugin=0;
   

   playerSettingsPlugin=(PlayerSettingsPlugin*) (Simulator::pluginManager.get("PlayerSettings"));
//    playerSettingsPlugin=0;
/*   cerr<<"Got Player Settings Plugin : "<<playerSettingsPlugin<<endl;
   cerr<<"numberOfLegendBoxes "<<playerSettingsPlugin->getPlayerSettings().numberOfLegendBoxes<<endl;
   cerr<<"numberOfLegendBoxesFlag "<<playerSettingsPlugin->getPlayerSettings().numberOfLegendBoxesFlag<<endl;*/


   if(playerSettingsPlugin){
      eventStart->playerSettings=playerSettingsPlugin->getPlayerSettings();
   }

   eventStart->xSize=fieldDim.x;
   eventStart->ySize=fieldDim.y;
   eventStart->zSize=fieldDim.z;
   eventStart->numSteps=simulator->getNumSteps();
   eventStart->latticeType=simulator->getPotts()->getLatticeType();
   cerr<<"eventStart->latticeType="<<eventStart->latticeType<<endl;
//    exit(0);
   
   cerr<<"This is getTargetObject="<<getTargetObject()<<endl;
//    QApplication::postEvent(getTargetObject(),eventStart);
//    cerr<<"This is getTargetWidget="<<getTargetWidget()<<endl;
   QApplication::postEvent(getTargetObject(),eventStart);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::postStartInit(){
   fillField3D(*graphFieldsPtr); ///fills field3D with values of the cell type
   fillConcentrationFields(*graphFieldsPtr, simulator->getConcentrationFieldNameMap());


   bufferFillFreeSemPtr->acquire();
   TransactionRefreshEvent *eventRefresh=new TransactionRefreshEvent();

   eventRefresh->mcStep=simulator->getStep();

   QApplication::postEvent(getTargetObject(),eventRefresh);
   bufferFillUsedSemPtr->release();



}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::loopWork(unsigned int _step)
{
   fillField3D(*graphFieldsPtr);
   fillConcentrationFields(*graphFieldsPtr, simulator->getConcentrationFieldNameMap());
   fillPressureFields(*graphFieldsPtr);
   fillVectorCellFields(*graphFieldsPtr);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DTransaction::loopWorkPostEvent(unsigned int _step){

   bufferFillFreeSemPtr->acquire();
   TransactionRefreshEvent *eventRefresh=new TransactionRefreshEvent(); ///once a pointer is owned by Qt object
                                                                        ///it will not create memory leak

   eventRefresh->mcStep=simulator->getStep();
   QApplication::postEvent(getTargetObject(),eventRefresh);
   bufferFillUsedSemPtr->release();

   pauseMutexPtr->lock();
   pauseMutexPtr->unlock();

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// QImage CC3DTransaction::apply(const QImage &image){
//    return image;
// }


void CC3DTransaction::simulationThreadCpp()
{
	simulationThreadPython();
	//try 
	//{
	//	// Load Plugin Libaries
	//	// Libaries in COMPUCELL3D_PLUGIN_PATH can override the
	//	// DEFAULT_PLUGIN_PATH

	//	char *steppablePath = getenv("COMPUCELL3D_STEPPABLE_PATH");
	//	cerr<<"steppablePath="<<steppablePath<<endl;
	//	if (steppablePath) Simulator::steppableManager.loadLibraries(steppablePath);

	//	char *pluginPath = getenv("COMPUCELL3D_PLUGIN_PATH");
	//	cerr<<"pluginPath="<<pluginPath<<endl;
	//	if (pluginPath) Simulator::pluginManager.loadLibraries(pluginPath);

	//	#ifdef DEFAULT_PLUGIN_PATH
	//	Simulator::pluginManager.loadLibraries(DEFAULT_PLUGIN_PATH);
	//	#endif

	//	BasicPluginManager<Plugin>::infos_t *infos = &Simulator::pluginManager.getPluginInfos();

	//	if (!infos->empty()) 
	//	{
	//		cerr << "Found the following plugins:" << endl;
	//		BasicPluginManager<Plugin>::infos_t::iterator it; 
	//		for (it = infos->begin(); it != infos->end(); it++)
	//		cerr << "  " << *(*it) << endl;
	//		cerr << endl;
	//	}


	//	// Create Simulator
	//	Simulator sim;
	//	simulator=&sim;

 //               graphFieldsPtr->clearAllocatedFields();
	//	// Initialize Xerces
	//	XMLPlatformUtils::Initialize();

	//	// Create parser
	//	BasicSmartPointer<XMLPullParser> parser = XMLPullParser::createInstance();

	//	// Parse config
	//	try 
	//	{
	//		parser->initParse(filename.c_str());
	//		parser->match(XMLEventTypes::START_DOCUMENT, -XMLEventTypes::TEXT);

	//		// Parse
	//		parser->assertName("CompuCell3D");
	//		parser->match(XMLEventTypes::START_ELEMENT);
	//		sim.readXML(*parser);
	//		parser->match(XMLEventTypes::END_ELEMENT, -XMLEventTypes::TEXT);

	//		// End
	//		parser->match(XMLEventTypes::END_DOCUMENT);
	//		sim.initializeCC3D();

	//	} 
	//	catch (BasicException e) 
	//	{
	//		throw BasicException("While parsing configuration!", parser->getLocation(), e);
	//	}

	//	sim.extraInit();///additional initialization after all plugins and steppables have been loaded and preinitialized
	//	///get dimension of the cell field
	//	preStartInit();   
	//	// Run simulation
	//	sim.start();

	//	//cerr << "CC3DTransaction::simulationThreadCpp() stopThread ADDRESS: " << pstopSimulation << "\t" << *pstopSimulation << "\n"; // TESTING THREADING 
	//	
	//	postStartInit();
	//	 
	//	for (unsigned int i = 1; i <= sim.getNumSteps(); i++)//(i <= sim.getNumSteps()) && (!*pstopSimulation); i++) ////
	//	{       
	//		if (*pstopSimulation) break;//return;

	//		sim.step(i);
	//		if(!( i % screenUpdateFrequency))
	//		{
	//			loopWork(i);
	//			loopWorkPostEvent(i);
	//		}
	//		//cerr << "CC3DTransaction::simulationThreadCpp(), FOR LOOP, stopThread ADDRESS: " << pstopSimulation << "\t" << *pstopSimulation << "\n"; // TESTING THREADING 
	//	}

	//	//if (*pstopSimulation) return;

	//	cerr<<"final: "<<sim.getNumSteps()<<endl;
	//	sim.finish();

	//	cerr<<"Done:"<<endl;
	//	return;
	//} 
	//catch (const XMLException &e) 
	//{
	//	cerr << "ERROR: " << XercesStr(e.getMessage()) << endl;
	//} 
	//catch (const BasicException &e) 
	//{
	//	cerr << "ERROR: " << e << endl;
	//}
}

void CC3DTransaction::simulationThreadPython(){
	string scriptName(pyDataConf.pythonFileName.toStdString());
	string path("");

	//    if(pyScriptRunner){
	//       pyScriptRunner->simulationPython(scriptName,path);
	//    }else{
	//       cerr<<"You need to register pyScript Runner object at the Python level"<<endl;
	//       exit(0);
	//    }
	cerr<<" \n\n\n\n BEFORE simulationPython\n\n\n "<<endl;
	try{
	   simulationPython(scriptName,path,this);
	}catch (const BasicException &e) {
		cerr<<"EXCEPTION FROM C++"<<endl;
		stringstream lineNumberString;
		lineNumberString<<e.getLocation().getLine();
		cerr << "ERROR: " << e.getMessage()+"\nLocation\n"+e.getLocation().getFilename() << "\nLine:\n"<<lineNumberString.str()<<endl;
		handleErrorMessage("Exception in C++ CompuCell3D code ", e.getMessage()+"\nLocation\n"+e.getLocation().getFilename() + "\nLine:\n"+lineNumberString.str());
	}
	

	cerr<<" \n\n\n\n AFTER simulationPython\n\n\n "<<endl;
   TransactionFinishEvent *eventFinish=new TransactionFinishEvent(); ///once a pointer is owned by Qt object
                                                                        ///it will not create memory leak

   eventFinish->message="FINISHED SIMULATION";
   QApplication::postEvent(getTargetObject(),eventFinish);
   



//    FILE *fp=fopen("script.py","r");
//    
//    Py_Initialize();
//    PyRun_SimpleFile(fp,"script.py");
//    Py_Finalize();
//    fclose(fp);
}


void CC3DTransaction::sendStopSimulationRequest(){
	TransactionStopSimulationEvent *eventStopSimulation=new TransactionStopSimulationEvent(); ///once a pointer is owned by Qt object
                                                                        ///it will not create memory leak

   eventStopSimulation->message="STOP SIMULATION REQUEST";
   QApplication::postEvent(getTargetObject(),eventStopSimulation);
}


void CC3DTransaction::applySimulation(bool *_stopSimulation)
{
	pstopSimulation = _stopSimulation; // Here is the ONLY place where pstopSimulation is assigned!

	cerr<<"runPythonFlag="<<runPythonFlag<<endl;
	cerr<<"useXMLFileFlag="<<useXMLFileFlag<<endl;
	if(runPythonFlag){
		simulationThreadPython();
	}else if(useXMLFileFlag){
		//cerr<<"*********************************************************************************************RUNNING CPP SIMULATION"<<endl;
		simulationThreadCpp();
	}else{
		cerr<<" No Simulation File has been specified"<<endl;
		exit(0);
	}
}

void CC3DTransaction::handleErrorMessage(std::string _errorCategory, std::string  _error){
   TransactionErrorEvent *eventError=new TransactionErrorEvent(); ///once a pointer is owned by Qt object
                                                                        ///it will not create memory leak	
	if(_errorCategory!="")
		eventError->errorCategory=QString(_errorCategory.c_str());
	if(_error!="")
		eventError->message=QString(_error.c_str());

   QApplication::postEvent(getTargetObject(),eventError);

}