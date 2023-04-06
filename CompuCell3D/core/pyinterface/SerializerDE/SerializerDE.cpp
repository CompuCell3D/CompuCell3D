#include <SerializerDE.h>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <pyinterface/PlayerPythonNew/FieldStorage.h>
#include <Logger/CC3DLogger.h>
#include <CompuCell3D/Automaton/Automaton.h> //to get type id to type name mapping
#include <Utils/Coordinates3D.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
//#include <vtkFloatArray.h>
#include <vtkCharArray.h>
#include <vtkLongArray.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkStructuredPointsReader.h>
#include <vtkPointData.h>

#include <map>
#include <iostream>
#include <fstream>


using namespace std;
using namespace CompuCell3D;

SerializerDE::SerializerDE():sim(0),potts(0),cellFieldG(0)
{}

SerializerDE::~SerializerDE(){}

void SerializerDE::init(Simulator * _sim){
	sim=_sim;
	if (!sim) {
		cout << "SerializerDE::init():  sim is null" << endl;
		exit(-1);
	}
	potts=sim->getPotts();
	if (!potts) {
		cout << "SerializerDE::init():  potts is null" << endl;
		exit(-1);
	}

    
	cellFieldG=potts->getCellFieldG();
	fieldDim=cellFieldG->getDim();
        
}

bool SerializerDE::serializeCellField(SerializeData &_sd){
	vtkStructuredPoints *fieldData=vtkStructuredPoints::New();
	fieldData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);
	CC3D_Log(LOG_DEBUG) << "fieldDim="<<fieldDim;
	CC3D_Log(LOG_DEBUG) << "potts="<<potts;

	vtkCharArray *typeArray=vtkCharArray::New();
	typeArray->SetName("CellType");

	
	vtkLongArray *idArray=vtkLongArray::New();
	idArray->SetName("CellId");


	vtkLongArray *clusterIdArray=vtkLongArray::New();
	clusterIdArray->SetName("ClusterId");



	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	typeArray->SetNumberOfValues(numberOfValues);
	idArray->SetNumberOfValues(numberOfValues);
	clusterIdArray->SetNumberOfValues(numberOfValues);

	Point3D pt;
	
	long offset=0;
	CellG* cell;
	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				cell=cellFieldG->get(pt);
				if(cell){
					typeArray->SetValue(offset,cell->type);
					idArray->SetValue(offset,cell->id);
					clusterIdArray->SetValue(offset,cell->clusterId);
				}else{
					typeArray->SetValue(offset,0);
					idArray->SetValue(offset,0);
					clusterIdArray->SetValue(offset,0);					
				}
				++offset;
			}
			fieldData->GetPointData()->AddArray(typeArray);
			fieldData->GetPointData()->AddArray(idArray);			
			fieldData->GetPointData()->AddArray(clusterIdArray);			

			typeArray->Delete();
			idArray->Delete();
			clusterIdArray->Delete();

    //writing structured points to the disk
	vtkStructuredPointsWriter * fieldDataWriter=vtkStructuredPointsWriter::New();
	fieldDataWriter->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	if (binaryFlag)
	    fieldDataWriter->SetFileTypeToBinary();
	else
	    fieldDataWriter->SetFileTypeToASCII();

#if defined(VTK6) || defined(VTK9)
    fieldDataWriter->SetInputData(fieldData);
#endif
#if !defined(VTK6) && !defined(VTK9)
    fieldDataWriter->SetInput(fieldData);
#endif
	
	fieldDataWriter->Write();
	fieldDataWriter->Delete();

	return true;

}

bool SerializerDE::loadCellField(SerializeData &_sd){

    //writing structured points to the disk
	vtkStructuredPointsReader * fieldDataReader=vtkStructuredPointsReader::New();
	fieldDataReader->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	//if (binaryFlag)
	//    fieldDataReader->SetFileTypeToBinary();
	//else
	//    fieldDataReader->SetFileTypeToASCII();
	fieldDataReader->Update();
	vtkStructuredPoints *fieldData=fieldDataReader->GetOutput();
    
    

	vtkCharArray *typeArray =(vtkCharArray *) fieldData->GetPointData()->GetArray("CellType");
	vtkLongArray *idArray = (vtkLongArray *) fieldData->GetPointData()->GetArray("CellId");
	vtkLongArray *clusterIdArray = (vtkLongArray *) fieldData->GetPointData()->GetArray("ClusterId");

	
	Point3D pt;
	
	long offset=0;
	CellG* cell;
	char type;
	long cellId;
	long clusterId;

	std::map<long, CellG*> existingCellsMap;
	std::map<long, CellG*>::iterator mitr;
	

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				type=typeArray->GetValue(offset);                
				if (!type){
					++offset;
					continue;
				}

				cellId=idArray->GetValue(offset);				
				clusterId=clusterIdArray->GetValue(offset);


				if ( existingCellsMap.find(cellId) != existingCellsMap.end() ){
					//reuse new cell
					cellFieldG->set(pt, existingCellsMap[cellId]);
					potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
					//It is necessary to do it this way because steppers are called only when we are performing pixel copies
					// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
					//inventory unless you call steppers(VolumeTrackerPlugin) explicitely

				}else{
					//create new cell
					cell = potts->createCellGSpecifiedIds(pt,cellId,clusterId);
					cell->type=type;
					potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
					//It is necessary to do it this way because steppers are called only when we are performing pixel copies
					// but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
					//inventory unless you call steppers(VolumeTrackerPlugin) explicitely
					existingCellsMap[cellId]=cell;

				}


				++offset;
			}

	fieldDataReader->Delete();

	return true;

}




bool SerializerDE::serializeConcentrationField(SerializeData &_sd){
	
	
	vtkStructuredPoints *fieldData=vtkStructuredPoints::New();
	fieldData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);
	

	Field3D<float> *fieldPtr=0; 
	std::map<std::string,Field3D<float>*> & fieldMap=sim->getConcentrationFieldNameMap();
	std::map<std::string,Field3D<float>*>::iterator mitr;
	mitr=fieldMap.find(_sd.objectName);
	if(mitr!=fieldMap.end()){
		fieldPtr=mitr->second;
	}

	
	if(!fieldPtr)
		return false;

	vtkDoubleArray *fieldArray=vtkDoubleArray::New();
	fieldArray->SetName(_sd.objectName.c_str());
	//arrayNameVec.push_back(_sd.objectName);	

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	fieldArray->SetNumberOfValues(numberOfValues);
	long offset=0;
	Point3D pt;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){

				fieldArray->SetValue(offset,fieldPtr->get(pt));
				++offset;
			}
	fieldData->GetPointData()->AddArray(fieldArray);
	fieldArray->Delete();


    //writing structured points to the disk
	vtkStructuredPointsWriter * fieldDataWriter=vtkStructuredPointsWriter::New();
	fieldDataWriter->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	if (binaryFlag)
	    fieldDataWriter->SetFileTypeToBinary();
	else
	    fieldDataWriter->SetFileTypeToASCII();
        
#if defined(VTK6) || defined(VTK9)
    fieldDataWriter->SetInputData(fieldData);
#endif
#if !defined(VTK6) && !defined(VTK9)
    fieldDataWriter->SetInput(fieldData);
#endif


    fieldDataWriter->Write();
	fieldDataWriter->Delete();

	return true;

}

bool SerializerDE::loadConcentrationField(SerializeData &_sd){

	Field3D<float> *fieldPtr=0; 
	std::map<std::string,Field3D<float>*> & fieldMap=sim->getConcentrationFieldNameMap();
	std::map<std::string,Field3D<float>*>::iterator mitr;
	mitr=fieldMap.find(_sd.objectName);
	if(mitr!=fieldMap.end()){
		fieldPtr=mitr->second;
	}
	
	if(!fieldPtr)
		return false;


    //writing structured points to the disk
	vtkStructuredPointsReader * fieldDataReader=vtkStructuredPointsReader::New();
	fieldDataReader->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	//if (binaryFlag)
	//    fieldDataReader->SetFileTypeToBinary();
	//else
	//    fieldDataReader->SetFileTypeToASCII();
	fieldDataReader->Update();
	vtkStructuredPoints *fieldData=fieldDataReader->GetOutput();

	vtkDoubleArray *fieldArray =(vtkDoubleArray *) fieldData->GetPointData()->GetArray(_sd.objectName.c_str());
	
	Point3D pt;
	
	long offset=0;
	double val;
	

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				val=fieldArray ->GetValue(offset);
				fieldPtr->set(pt,val);
				++offset;
			}

	fieldDataReader->Delete();

	return true;

}



bool SerializerDE::serializeScalarField(SerializeData &_sd){
	if (!_sd.objectPtr)
		return false;

	FieldStorage::floatField3D_t * fieldPtr=(FieldStorage::floatField3D_t *) _sd.objectPtr;

	vtkStructuredPoints *fieldData=vtkStructuredPoints::New();
	fieldData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);
	


	vtkDoubleArray *fieldArray=vtkDoubleArray::New();
	fieldArray->SetName(_sd.objectName.c_str());
	//arrayNameVec.push_back(_sd.objectName);	

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	fieldArray->SetNumberOfValues(numberOfValues);
	long offset=0;
	Point3D pt;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){

				fieldArray->SetValue(offset,(*fieldPtr)[pt.x][pt.y][pt.z]);
				++offset;
			}
	fieldData->GetPointData()->AddArray(fieldArray);
	fieldArray->Delete();


    //writing structured points to the disk
	vtkStructuredPointsWriter * fieldDataWriter=vtkStructuredPointsWriter::New();
	fieldDataWriter->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	if (binaryFlag)
	    fieldDataWriter->SetFileTypeToBinary();
	else
	    fieldDataWriter->SetFileTypeToASCII();
#if defined(VTK6) || defined(VTK9)
    fieldDataWriter->SetInputData(fieldData);
#endif
#if !defined(VTK6) && !defined(VTK9)
    fieldDataWriter->SetInput(fieldData);
#endif


    fieldDataWriter->Write();
	fieldDataWriter->Delete();

	return true;


	
}


bool SerializerDE::loadScalarField(SerializeData &_sd){
	if (!_sd.objectPtr)
		return false;

	FieldStorage::floatField3D_t * fieldPtr=(FieldStorage::floatField3D_t *) _sd.objectPtr;


	vtkStructuredPointsReader * fieldDataReader=vtkStructuredPointsReader::New();
	fieldDataReader->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	//if (binaryFlag)
	//    fieldDataReader->SetFileTypeToBinary();
	//else
	//    fieldDataReader->SetFileTypeToASCII();
	fieldDataReader->Update();
	vtkStructuredPoints *fieldData=fieldDataReader->GetOutput();

	vtkDoubleArray *fieldArray =(vtkDoubleArray *) fieldData->GetPointData()->GetArray(_sd.objectName.c_str());
	
	Point3D pt;
	
	long offset=0;
	double val;
	

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				val=fieldArray ->GetValue(offset);
				(*fieldPtr)[pt.x][pt.y][pt.z]=val;				
				++offset;
			}

	fieldDataReader->Delete();

	return true;



	
}


bool SerializerDE::serializeScalarFieldCellLevel(SerializeData &_sd){
	if (!_sd.objectPtr)
		return false;

	FieldStorage::scalarFieldCellLevel_t *  fieldPtr=(FieldStorage::scalarFieldCellLevel_t * ) _sd.objectPtr;

	vtkStructuredPoints *fieldData=vtkStructuredPoints::New();
	fieldData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);
	
	vtkDoubleArray *fieldArray=vtkDoubleArray::New();
	fieldArray->SetName(_sd.objectName.c_str());
	//arrayNameVec.push_back(_sd.objectName);	

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	fieldArray->SetNumberOfValues(numberOfValues);

	FieldStorage::scalarFieldCellLevel_t::iterator mitr;

	long offset=0;
	Point3D pt;
	CellG * cell;
	float con;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				cell=cellFieldG->get(pt);
				if (cell){
					mitr=fieldPtr->find(cell);
					if(mitr!=fieldPtr->end()){
						con=mitr->second;
					}else{
						con=0.0;
					}
				}else{
					con=0.0;
				}
				fieldArray->SetValue(offset,con);
				++offset;

			}
	fieldData->GetPointData()->AddArray(fieldArray);
	fieldArray->Delete();


    //writing structured points to the disk
	vtkStructuredPointsWriter * fieldDataWriter=vtkStructuredPointsWriter::New();
	fieldDataWriter->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	if (binaryFlag)
	    fieldDataWriter->SetFileTypeToBinary();
	else
	    fieldDataWriter->SetFileTypeToASCII();
#if defined(VTK6) || defined(VTK9)
    fieldDataWriter->SetInputData(fieldData);
#endif
#if !defined(VTK6) && !defined(VTK9)
    fieldDataWriter->SetInput(fieldData);
#endif


    fieldDataWriter->Write();
	fieldDataWriter->Delete();

	return true;

}


bool SerializerDE::loadScalarFieldCellLevel(SerializeData &_sd){
	if (!_sd.objectPtr)
		return false;

	FieldStorage::scalarFieldCellLevel_t *  fieldPtr=(FieldStorage::scalarFieldCellLevel_t * ) _sd.objectPtr;


	vtkStructuredPointsReader * fieldDataReader=vtkStructuredPointsReader::New();
	fieldDataReader->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	//if (binaryFlag)
	//    fieldDataReader->SetFileTypeToBinary();
	//else
	//    fieldDataReader->SetFileTypeToASCII();
	fieldDataReader->Update();
	vtkStructuredPoints *fieldData=fieldDataReader->GetOutput();

	vtkDoubleArray *fieldArray =(vtkDoubleArray *) fieldData->GetPointData()->GetArray(_sd.objectName.c_str());
	Point3D pt;

	FieldStorage::scalarFieldCellLevel_t::iterator mitr;
	long offset=0;
	double val;
	CellG * cell;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				val=fieldArray ->GetValue(offset);
				cell=cellFieldG->get(pt);
				if (cell){
					mitr=fieldPtr->find(cell);
					if(mitr!=fieldPtr->end()){
						;
					}else{
						fieldPtr->insert(make_pair(cell,val));
						
					}
				}
				++offset;
			}

	fieldDataReader->Delete();

	return true;



	
}


bool SerializerDE::serializeVectorField(SerializeData &_sd){
	if (!_sd.objectPtr)
		return false;

	FieldStorage::vectorField3D_t *  fieldPtr=(FieldStorage::vectorField3D_t * ) _sd.objectPtr;

	vtkStructuredPoints *fieldData=vtkStructuredPoints::New();
	fieldData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);
	


	vtkDoubleArray *fieldArray=vtkDoubleArray::New();
	fieldArray->SetNumberOfComponents(3); // we will store here 3 component vectors, not scalars	
	fieldArray->SetName(_sd.objectName.c_str());
	//arrayNameVec.push_back(_sd.objectName);	

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	fieldArray->SetNumberOfTuples(numberOfValues);
	long offset=0;
	Point3D pt;
// 	Coordinates3D<float> vecTmp;
        float x,y,z;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
// 				vecTmp=(*fieldPtr)[pt.x][pt.y][pt.z];
                                x=(*fieldPtr)[pt.x][pt.y][pt.z][0];
                                y=(*fieldPtr)[pt.x][pt.y][pt.z][1];
                                z=(*fieldPtr)[pt.x][pt.y][pt.z][2];
                                fieldArray->SetTuple3(offset,x,y,z);
								CC3D_Log(LOG_TRACE) << "vec=" << x << ", " << y << ", " << z;
// 				fieldArray->SetTuple3(offset,vecTmp.x,vecTmp.y,vecTmp.z);
				++offset;

			}
	fieldData->GetPointData()->AddArray(fieldArray);
	fieldArray->Delete();


    //writing structured points to the disk
	vtkStructuredPointsWriter * fieldDataWriter=vtkStructuredPointsWriter::New();
	fieldDataWriter->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	if (binaryFlag)
	    fieldDataWriter->SetFileTypeToBinary();
	else
	    fieldDataWriter->SetFileTypeToASCII();

#if defined(VTK6) || defined(VTK9)
    fieldDataWriter->SetInputData(fieldData);
#endif
#if !defined(VTK6) && !defined(VTK9)
    fieldDataWriter->SetInput(fieldData);
#endif


    fieldDataWriter->Write();
	fieldDataWriter->Delete();

	return true;

}

bool SerializerDE::loadVectorField(SerializeData &_sd){
	if (!_sd.objectPtr)
		return false;

	FieldStorage::vectorField3D_t *  fieldPtr=(FieldStorage::vectorField3D_t * ) _sd.objectPtr;


	vtkStructuredPointsReader * fieldDataReader=vtkStructuredPointsReader::New();
	fieldDataReader->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	//if (binaryFlag)
	//    fieldDataReader->SetFileTypeToBinary();
	//else
	//    fieldDataReader->SetFileTypeToASCII();
	fieldDataReader->Update();
	vtkStructuredPoints *fieldData=fieldDataReader->GetOutput();

	vtkDoubleArray *fieldArray =(vtkDoubleArray *) fieldData->GetPointData()->GetArray(_sd.objectName.c_str());

	
	long offset=0;
	Point3D pt;
	
	double tuple[3];

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){

				fieldArray->GetTypedTuple(offset,tuple);

// 				(*fieldPtr)[pt.x][pt.y][pt.z]=Coordinates3D<float>(tuple[0],tuple[1],tuple[2]) ;
                                
                                (*fieldPtr)[pt.x][pt.y][pt.z][0]=tuple[0];
                                (*fieldPtr)[pt.x][pt.y][pt.z][1]=tuple[1];
                                (*fieldPtr)[pt.x][pt.y][pt.z][2]=tuple[2];
				++offset;

			}
	

	fieldDataReader->Delete();
	return true;

}


bool SerializerDE::serializeVectorFieldCellLevel(SerializeData &_sd){
	if (!_sd.objectPtr)
		return false;

	FieldStorage::vectorFieldCellLevel_t *  fieldPtr=(FieldStorage::vectorFieldCellLevel_t * ) _sd.objectPtr;

	vtkStructuredPoints *fieldData=vtkStructuredPoints::New();
	fieldData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);
	


	vtkDoubleArray *fieldArray=vtkDoubleArray::New();
	fieldArray->SetNumberOfComponents(3); // we will store here 3 component vectors, not scalars	
	fieldArray->SetName(_sd.objectName.c_str());
	//arrayNameVec.push_back(_sd.objectName);	

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	fieldArray->SetNumberOfTuples(numberOfValues);
	long offset=0;
	Point3D pt;
	Coordinates3D<float> vecTmp;
	FieldStorage::vectorFieldCellLevelItr_t mitr;

	CellG * cell;


	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				cell=cellFieldG->get(pt);
				if (cell){
					mitr=fieldPtr->find(cell);
					if(mitr!=fieldPtr->end()){
						vecTmp=mitr->second;
						fieldArray->SetTuple3(offset,vecTmp.x,vecTmp.y,vecTmp.z);
					}else{
						//vecTmp=Coordinates3D<float>();
					}
				}else{
					//vecTmp=Coordinates3D<float>();
				}

				++offset;

			}
	fieldData->GetPointData()->AddArray(fieldArray);
	fieldArray->Delete();

    //writing structured points to the disk
	vtkStructuredPointsWriter * fieldDataWriter=vtkStructuredPointsWriter::New();
	fieldDataWriter->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	if (binaryFlag)
	    fieldDataWriter->SetFileTypeToBinary();
	else
	    fieldDataWriter->SetFileTypeToASCII();
#if defined(VTK6) || defined(VTK9)
    fieldDataWriter->SetInputData(fieldData);
#endif
#if !defined(VTK6) && !defined(VTK9)
    fieldDataWriter->SetInput(fieldData);
#endif


    fieldDataWriter->Write();
	fieldDataWriter->Delete();

	return true;


}

bool SerializerDE::loadVectorFieldCellLevel(SerializeData &_sd){
	if (!_sd.objectPtr)
		return false;

	FieldStorage::vectorFieldCellLevel_t *  fieldPtr=(FieldStorage::vectorFieldCellLevel_t * ) _sd.objectPtr;

	vtkStructuredPointsReader * fieldDataReader=vtkStructuredPointsReader::New();
	fieldDataReader->SetFileName(_sd.fileName.c_str());

	bool binaryFlag=(_sd.fileFormat=="binary");	

	//if (binaryFlag)
	//    fieldDataReader->SetFileTypeToBinary();
	//else
	//    fieldDataReader->SetFileTypeToASCII();
	fieldDataReader->Update();
	vtkStructuredPoints *fieldData=fieldDataReader->GetOutput();

	vtkDoubleArray *fieldArray =(vtkDoubleArray *) fieldData->GetPointData()->GetArray(_sd.objectName.c_str());


	long offset=0;
	Point3D pt;

	FieldStorage::vectorFieldCellLevelItr_t mitr;

	CellG * cell;
	double tuple[3];

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)	
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){

				cell=cellFieldG->get(pt);
				if (cell){
					
					mitr=fieldPtr->find(cell);
					if(mitr!=fieldPtr->end()){
						;
					}else{
						fieldArray->GetTypedTuple(offset,tuple);
						CC3D_Log(LOG_DEBUG) << "inserting "<<Coordinates3D<float>(tuple[0],tuple[1],tuple[2]);
						fieldPtr->insert(make_pair(cell,Coordinates3D<float>(tuple[0],tuple[1],tuple[2])));
						
					}
				}

			++offset;
			}

	fieldDataReader->Delete();
	return true;
}