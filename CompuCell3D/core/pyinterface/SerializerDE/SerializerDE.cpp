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
#include <vtkFloatArray.h>
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

SerializerDE::SerializerDE() : sim(0), potts(0), cellFieldG(0) {
    initializeSerializeConcentrationFunctionMap();
    initializeLoadConcentrationFunctionMap();
}

SerializerDE::~SerializerDE() {}

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

void SerializerDE::initializeSerializeConcentrationFunctionMap() {
    serializeConcentrationFunctionMap = {
            {typeid(char), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<char>(_sd, static_cast<Field3D<char>*>(ptr));
            }},
            {typeid(unsigned char), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<unsigned char>(_sd, static_cast<Field3D<unsigned char>*>(ptr));
            }},
            {typeid(short), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<short>(_sd, static_cast<Field3D<short>*>(ptr));
            }},
            {typeid(unsigned short), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<unsigned short>(_sd, static_cast<Field3D<unsigned short>*>(ptr));
            }},
            {typeid(int), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<int>(_sd, static_cast<Field3D<int>*>(ptr));
            }},
            {typeid(unsigned int), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<unsigned int>(_sd, static_cast<Field3D<unsigned int>*>(ptr));
            }},
            {typeid(long), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<long>(_sd, static_cast<Field3D<long>*>(ptr));
            }},
            {typeid(unsigned long), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<unsigned long>(_sd, static_cast<Field3D<unsigned long>*>(ptr));
            }},
            {typeid(float), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<float>(_sd, static_cast<Field3D<float>*>(ptr));
            }},
            {typeid(double), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<double>(_sd, static_cast<Field3D<double>*>(ptr));
            }},
            {typeid(long long), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<long long>(_sd, static_cast<Field3D<long long>*>(ptr));
            }},
            {typeid(unsigned long long), [this](SerializeData& _sd, void* ptr) {
                return this->serializeConcentrationFieldTyped<unsigned long long>(_sd, static_cast<Field3D<unsigned long long>*>(ptr));
            }}
    };
}


void SerializerDE::initializeLoadConcentrationFunctionMap() {
    loadConcentrationFunctionMap = {
            {typeid(char), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<char>(_sd, static_cast<Field3D<char>*>(ptr));
            }},
            {typeid(unsigned char), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<unsigned char>(_sd, static_cast<Field3D<unsigned char>*>(ptr));
            }},
            {typeid(short), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<short>(_sd, static_cast<Field3D<short>*>(ptr));
            }},
            {typeid(unsigned short), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<unsigned short>(_sd, static_cast<Field3D<unsigned short>*>(ptr));
            }},
            {typeid(int), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<int>(_sd, static_cast<Field3D<int>*>(ptr));
            }},
            {typeid(unsigned int), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<unsigned int>(_sd, static_cast<Field3D<unsigned int>*>(ptr));
            }},
            {typeid(long), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<long>(_sd, static_cast<Field3D<long>*>(ptr));
            }},
            {typeid(unsigned long), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<unsigned long>(_sd, static_cast<Field3D<unsigned long>*>(ptr));
            }},
            {typeid(float), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<float>(_sd, static_cast<Field3D<float>*>(ptr));
            }},
            {typeid(double), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<double>(_sd, static_cast<Field3D<double>*>(ptr));
            }},
            {typeid(long long), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<long long>(_sd, static_cast<Field3D<long long>*>(ptr));
            }},
            {typeid(unsigned long long), [this](SerializeData& _sd, void* ptr) {
                return this->loadConcentrationFieldTyped<unsigned long long>(_sd, static_cast<Field3D<unsigned long long>*>(ptr));
            }}
    };
}



std::tuple<std::type_index, void*> SerializerDE::getFieldTypeAndPointer( const std::string& fieldName) {
    Field3DTypeBase* conFieldBasePtr = sim->getGenericScalarFieldTypeBase(fieldName);


    static const std::unordered_map<std::type_index, std::function<void*(Field3DTypeBase*)>> typeCaster = {
            {typeid(char), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<char>*>(base)); }},
            {typeid(unsigned char), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned char>*>(base)); }},
            {typeid(short), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<short>*>(base)); }},
            {typeid(unsigned short), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned short>*>(base)); }},
            {typeid(int), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<int>*>(base)); }},
            {typeid(unsigned int), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned int>*>(base)); }},
            {typeid(long), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<long>*>(base)); }},
            {typeid(unsigned long), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned long>*>(base)); }},
            {typeid(long long), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<long long>*>(base)); }},
            {typeid(unsigned long long), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned long long>*>(base)); }},
            {typeid(float), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<float>*>(base)); }},
            {typeid(double), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<double>*>(base)); }},
            {typeid(long double), [](Field3DTypeBase* base) { return static_cast<void*>(dynamic_cast<NumpyArrayWrapper3DImpl<long double>*>(base)); }},
    };
    if (conFieldBasePtr) {
        std::type_index fieldType = conFieldBasePtr->getType();
        auto it = typeCaster.find(fieldType);

        if (it != typeCaster.end()) {
            return {fieldType, it->second(conFieldBasePtr)};
        }

    }

    Field3D<float> *conFieldPtr = nullptr;
    std::map<std::string, Field3D<float> *> &fieldMap = sim->getConcentrationFieldNameMap();
    std::map<std::string, Field3D<float> *>::iterator mitr;
    mitr = fieldMap.find(fieldName);
    if (mitr != fieldMap.end()) {
        conFieldPtr = mitr->second;
    }

    if (conFieldPtr){

        return {std::type_index(typeid(float)), conFieldPtr};
    }


    return {std::type_index(typeid(void)), nullptr};  // Unsupported type
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



template<typename T>
bool SerializerDE::serializeConcentrationFieldTyped(SerializeData &_sd, Field3D<T> *fieldPtr){

    vtkStructuredPoints *fieldData=vtkStructuredPoints::New();
    fieldData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);

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


bool SerializerDE::serializeConcentrationField(SerializeData &_sd){

    // Retrieve field type and pointer
    auto result = getFieldTypeAndPointer(_sd.objectName);
    std::type_index fieldType = std::get<0>(result);
    void* fieldPtr = std::get<1>(result);

    if (!fieldPtr || fieldType == typeid(void)) {
        return false;
    }


    // Look up the function in serializeConcentrationFunctionMap

    auto it = serializeConcentrationFunctionMap.find(fieldType);
    if (it != serializeConcentrationFunctionMap.end()) {
        return it->second(_sd, fieldPtr);
    }

    return false;



}

template<typename T>
bool SerializerDE::loadConcentrationFieldTyped(SerializeData &_sd, Field3D<T> *fieldPtr){

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


bool SerializerDE::loadConcentrationField(SerializeData &_sd){

    // Retrieve field type and pointer
    auto result = getFieldTypeAndPointer(_sd.objectName);
    std::type_index fieldType = std::get<0>(result);
    void* fieldPtr = std::get<1>(result);

    if (!fieldPtr || fieldType == typeid(void)) {
        return false;
    }

    // Look up the function in loadConcentrationFunctionMap
    auto it = loadConcentrationFunctionMap.find(fieldType);
    if (it != loadConcentrationFunctionMap.end()) {
        return it->second(_sd, fieldPtr);
    }

    return false;

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


bool SerializerDE::serializeSharedVectorFieldNumpy(SerializeData &_sd){

    Simulator::vectorField3DNumpyImpl_t * fieldPtr = nullptr;
    std::map<std::string,Simulator::vectorField3DNumpyImpl_t*> vectorFieldMap=sim->getVectorFieldMap();
    auto mitr=vectorFieldMap.find(_sd.objectName);
    if(mitr != vectorFieldMap.end()){
        fieldPtr = mitr->second;
    }

    if(!fieldPtr)
        return false;


    vtkStructuredPoints *fieldData=vtkStructuredPoints::New();
    fieldData->SetDimensions(fieldDim.x,fieldDim.y,fieldDim.z);



    vtkFloatArray *fieldArray=vtkFloatArray::New();
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
                auto vec =  fieldPtr->get(pt);
// 				vecTmp=(*fieldPtr)[pt.x][pt.y][pt.z];
                fieldArray->SetTuple3(offset,vec.x,vec.y,vec.z);
//                CC3D_Log(LOG_TRACE) << "vec=" << x << ", " << y << ", " << z;
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

bool SerializerDE::loadSharedVectorFieldNumpy(SerializeData &_sd) {

    Simulator::vectorField3DNumpyImpl_t * fieldPtr = nullptr;
    std::map<std::string,Simulator::vectorField3DNumpyImpl_t*> vectorFieldMap=sim->getVectorFieldMap();
    auto mitr=vectorFieldMap.find(_sd.objectName);
    if(mitr != vectorFieldMap.end()){
        fieldPtr = mitr->second;
    }

    if(!fieldPtr)
        return false;

    vtkStructuredPointsReader * fieldDataReader=vtkStructuredPointsReader::New();
    fieldDataReader->SetFileName(_sd.fileName.c_str());

    bool binaryFlag=(_sd.fileFormat=="binary");

    //if (binaryFlag)
    //    fieldDataReader->SetFileTypeToBinary();
    //else
    //    fieldDataReader->SetFileTypeToASCII();
    fieldDataReader->Update();
    vtkStructuredPoints *fieldData=fieldDataReader->GetOutput();

    auto *fieldArray =(vtkFloatArray *) fieldData->GetPointData()->GetArray(_sd.objectName.c_str());


    long offset=0;
    Point3D pt;

    float tuple[3];

    for(pt.z =0 ; pt.z<fieldDim.z ; ++pt.z)
        for(pt.y =0 ; pt.y<fieldDim.y ; ++pt.y)
            for(pt.x =0 ; pt.x<fieldDim.x ; ++pt.x){

                fieldArray->GetTypedTuple(offset,tuple);
                fieldPtr->set(pt, Coordinates3D<float>(tuple[0], tuple[1], tuple[2]));
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