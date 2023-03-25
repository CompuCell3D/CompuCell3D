#include "FieldWriter.h"
#include "CellGraphicsData.h"
#include <iostream>
#include <fstream>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Automaton/Automaton.h> //to get type id to type name mapping
#include <Logger/CC3DLogger.h>
#include <Utils/Coordinates3D.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkCharArray.h>
#include <vtkLongArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkType.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkStructuredPointsReader.h>
#include <vtkPointData.h>
#include <algorithm>
#include <cmath>

#include <vtkPythonUtil.h>

using namespace std;
using namespace CompuCell3D;

FieldWriter::FieldWriter():fsPtr(0),potts(0),sim(0),latticeData(0)
{

}
////////////////////////////////////////////////////////////////////////////////
FieldWriter::~FieldWriter(){
	if (latticeData){
		latticeData->Delete();
	}
}
////////////////////////////////////////////////////////////////////////////////
void FieldWriter::init(Simulator * _sim){

	sim=_sim;
	if (!sim) {
		cout << "FieldWriter::init():  sim is null" << endl;
		exit(-1);
	}
	potts=sim->getPotts();
	if (!potts) {
		cout << "FieldWriter::init():  potts is null" << endl;
		exit(-1);
	}
    latticeData=vtkStructuredPoints::New();
	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

    latticeData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);
}
////////////////////////////////////////////////////////////////////////////////
void FieldWriter::clear(){
	for (int i =0 ; i < arrayNameVec.size() ;++i){
		latticeData->GetPointData()->RemoveArray(arrayNameVec[i].c_str());
	}
	arrayNameVec.clear();
}
////////////////////////////////////////////////////////////////////////////////
void FieldWriter::setFileTypeToBinary(bool flag){
	binaryFlag = flag;
}
////////////////////////////////////////////////////////////////////////////////
void FieldWriter::writeFields(std::string _fileName){

	//latticeData->Print(cerr);
	vtkStructuredPointsWriter * latticeDataWriter=vtkStructuredPointsWriter::New();
	latticeDataWriter->SetFileName(_fileName.c_str());

	//get new field dim before each write event - in case simulation dimensions have changed
	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

    latticeData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);


//	if (binaryFlag)
//	    latticeDataWriter->SetFileTypeToBinary();
//	else
//	    latticeDataWriter->SetFileTypeToASCII();
        #if defined(VTK6) || defined(VTK9)
            latticeDataWriter->SetInputData(latticeData);
        #endif
        #if !defined(VTK6) && !defined(VTK9)
            latticeDataWriter->SetInput(latticeData);
        #endif
	int dim[3];
	latticeData->GetDimensions(dim);
	CC3D_Log(LOG_TRACE) << "dim 0="<<dim[0]<<" dim 1="<<dim[1]<<" dim 2="<<dim[2];

	latticeDataWriter->Write();
	latticeDataWriter->Delete();
}
////////////////////////////////////////////////////////////////////////////////
void FieldWriter::generatePIFFileFromVTKOutput(std::string _vtkFileName,std::string _pifFileName,short _dimX, short _dimY, short _dimZ,std::map<int,std::string> &typeIdTypeNameMap){

	vtkStructuredPointsReader * latticeDataReader=vtkStructuredPointsReader::New();
	latticeDataReader->SetFileName(_vtkFileName.c_str());
	latticeDataReader->Update();
	vtkStructuredPoints * lds=latticeDataReader->GetOutput();

	vtkCharArray *typeArrayRead=(vtkCharArray *)lds->GetPointData()->GetArray("CellType");
	vtkLongArray *idArrayRead=(vtkLongArray *)lds->GetPointData()->GetArray("CellId");
	vtkLongArray *clusterIdArrayRead=(vtkLongArray *)lds->GetPointData()->GetArray("ClusterId");

	ofstream outFile(_pifFileName.c_str());
	outFile<<"Include Clusters"<<endl;
	long offset=0;
	long id=0,clusterId=0;
	int type=0;

	for (int z=0 ; z<_dimZ ; ++z)
		for (int y=0 ; y<_dimY ; ++y)
			for (int x=0 ; x<_dimX ; ++x){
				type=typeArrayRead->GetValue(offset);
				if(!type){

				}else{
					clusterId=clusterIdArrayRead->GetValue(offset);
					id=idArrayRead->GetValue(offset);

					outFile<<clusterId<<"\t"<<id<<"\t"<<typeIdTypeNameMap[type]<<"\t"
					<<x<<"\t"<<x<<"\t"
					<<y<<"\t"<<y<<"\t"
					<<z<<"\t"<<z
					<<endl;
				}
				++offset;
			}
	latticeDataReader->Delete();
}
////////////////////////////////////////////////////////////////////////////////
void FieldWriter::generatePIFFileFromCurrentStateOfSimulation(std::string _pifFileName){

	ofstream outFile(_pifFileName.c_str());
	outFile<<"Include Clusters"<<endl;

	long id=0,clusterId=0;
	char type=0;

	CellG * cell;
	Point3D pt;
	Automaton *automaton=potts->getAutomaton();

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				cell=cellFieldG->get(pt);

				if(cell){
					type=cell->type;
					id=cell->id;
					clusterId=cell->clusterId;
					outFile<<clusterId<<"    "<<id<<"    "<<automaton->getTypeName(type)<<"    "
					<<pt.x<<"    "<<pt.x<<"    "
					<<pt.y<<"    "<<pt.y<<"    "
					<<pt.z<<"    "<<pt.z
					<<endl;
				}

			}
}
////////////////////////////////////////////////////////////////////////////////
void FieldWriter::addCellFieldForOutput(){
	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	vtkCharArray *typeArray=vtkCharArray::New();
	typeArray->SetName("CellType");
	arrayNameVec.push_back("CellType");

	vtkLongArray *idArray=vtkLongArray::New();
	idArray->SetName("CellId");
	arrayNameVec.push_back("CellId");

	vtkLongArray *clusterIdArray=vtkLongArray::New();
	clusterIdArray->SetName("ClusterId");
	arrayNameVec.push_back("ClusterId");

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
			latticeData->GetPointData()->AddArray(typeArray);
			latticeData->GetPointData()->AddArray(idArray);
			latticeData->GetPointData()->AddArray(clusterIdArray);

			typeArray->Delete();
			idArray->Delete();
			clusterIdArray->Delete();
}
////////////////////////////////////////////////////////////////////////////////
bool FieldWriter::addConFieldForOutput(std::string _conFieldName){
	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	Field3D<float> *conFieldPtr=0;
	std::map<std::string,Field3D<float>*> & fieldMap=sim->getConcentrationFieldNameMap();
	std::map<std::string,Field3D<float>*>::iterator mitr;
	mitr=fieldMap.find(_conFieldName);
	if(mitr!=fieldMap.end()){
		conFieldPtr=mitr->second;
	}

	if(!conFieldPtr)
		return false;

	vtkDoubleArray *conArray=vtkDoubleArray::New();
	conArray->SetName(_conFieldName.c_str());
	arrayNameVec.push_back(_conFieldName);

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	conArray->SetNumberOfValues(numberOfValues);
	long offset=0;
	Point3D pt;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				conArray->SetValue(offset,conFieldPtr->get(pt));
				++offset;
			}
	latticeData->GetPointData()->AddArray(conArray);
	conArray->Delete();
	return true;
}
////////////////////////////////////////////////////////////////////////////////
bool FieldWriter::addScalarFieldForOutput(std::string _scalarFieldName){
	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	FieldStorage::floatField3D_t * conFieldPtr=fsPtr->getScalarFieldByName(_scalarFieldName);

	if(!conFieldPtr)
		return false;

	vtkDoubleArray *conArray=vtkDoubleArray::New();
	conArray->SetName(_scalarFieldName.c_str());
	arrayNameVec.push_back(_scalarFieldName);

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	conArray->SetNumberOfValues(numberOfValues);
	long offset=0;
	Point3D pt;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				conArray->SetValue(offset,(*conFieldPtr)[pt.x][pt.y][pt.z]);
				++offset;
			}
	latticeData->GetPointData()->AddArray(conArray);
	conArray->Delete();
	return true;
}
////////////////////////////////////////////////////////////////////////////////
bool FieldWriter::addScalarFieldCellLevelForOutput(std::string _scalarFieldCellLevelName){

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();
	FieldStorage::scalarFieldCellLevel_t * conFieldPtr=fsPtr->getScalarFieldCellLevelFieldByName(_scalarFieldCellLevelName);

	if(!conFieldPtr)
		return false;

	FieldStorage::scalarFieldCellLevel_t::iterator mitr;

	vtkDoubleArray *conArray=vtkDoubleArray::New();
	conArray->SetName(_scalarFieldCellLevelName.c_str());
	arrayNameVec.push_back(_scalarFieldCellLevelName);

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	conArray->SetNumberOfValues(numberOfValues);
	long offset=0;
	Point3D pt;
	CellG * cell;
	float con;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
				cell=cellFieldG->get(pt);
				if (cell){
					mitr=conFieldPtr->find(cell);
					if(mitr!=conFieldPtr->end()){
						con=mitr->second;
					}else{
						con=0.0;
					}
				}else{
					con=0.0;
				}
				conArray->SetValue(offset,con);
				++offset;
			}
	latticeData->GetPointData()->AddArray(conArray);
	conArray->Delete();
	return true;
}
////////////////////////////////////////////////////////////////////////////////
bool FieldWriter::addVectorFieldForOutput(std::string _vectorFieldName){

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();

	FieldStorage::vectorField3D_t * vecFieldPtr=fsPtr->getVectorFieldFieldByName(_vectorFieldName);

	if(!vecFieldPtr)
		return false;

	vtkDoubleArray *vecArray=vtkDoubleArray::New();
	vecArray->SetNumberOfComponents(3); // we will store here 3 component vectors, not scalars
	vecArray->SetName(_vectorFieldName.c_str());
	arrayNameVec.push_back(_vectorFieldName);

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	vecArray->SetNumberOfTuples(numberOfValues); // if using more than one component data you need to use SetNumberOfComponent followed by SetNumberOfTuples
	long offset=0;
	Point3D pt;
	float x,y,z;
	Coordinates3D<float> vecTmp;

	for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)
		for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
			for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){
// 				vecTmp=(*vecFieldPtr)[pt.x][pt.y][pt.z];
                                x=(*vecFieldPtr)[pt.x][pt.y][pt.z][0];
                                y=(*vecFieldPtr)[pt.x][pt.y][pt.z][1];
                                z=(*vecFieldPtr)[pt.x][pt.y][pt.z][2];
				CC3D_Log(LOG_TRACE) << "vecTmp="<<vecTmp;
				vecArray->SetTuple3(offset,x,y,z);
				++offset;
			}

	latticeData->GetPointData()->AddArray(vecArray);
	vecArray->Delete();
	return true;
}
////////////////////////////////////////////////////////////////////////////////
bool FieldWriter::addVectorFieldCellLevelForOutput(std::string _vectorFieldCellLevelName){

	Field3D<CellG*> * cellFieldG=potts->getCellFieldG();
	Dim3D fieldDim=cellFieldG->getDim();
	FieldStorage::vectorFieldCellLevel_t * vecFieldPtr=fsPtr->getVectorFieldCellLevelFieldByName(_vectorFieldCellLevelName);

	if(!vecFieldPtr)
		return false;

	vtkDoubleArray *vecArray=vtkDoubleArray::New();
	vecArray->SetNumberOfComponents(3); // we will store here 3 component vectors, not scalars
	vecArray->SetName(_vectorFieldCellLevelName.c_str());
	arrayNameVec.push_back(_vectorFieldCellLevelName);

	long numberOfValues=fieldDim.x*fieldDim.y*fieldDim.z;

	vecArray->SetNumberOfTuples(numberOfValues);
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
					mitr=vecFieldPtr->find(cell);
					if(mitr!=vecFieldPtr->end()){
						vecTmp=mitr->second;
					}else{
						vecTmp=Coordinates3D<float>();

					}
				}else{
					vecTmp=Coordinates3D<float>();
				}
				vecArray->SetTuple3(offset,vecTmp.x,vecTmp.y,vecTmp.z);
				++offset;
			}
	latticeData->GetPointData()->AddArray(vecArray);
	vecArray->Delete();
	return true;
}
