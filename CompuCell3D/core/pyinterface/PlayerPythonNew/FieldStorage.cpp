#include "FieldStorage.h"
#include <iostream>

using namespace std;
using namespace CompuCell3D;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::FieldStorage(){

}

void FieldStorage::setDim(Dim3D _dim){fieldDim=_dim;}

Dim3D FieldStorage::getDim(){return fieldDim;}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 void FieldStorage::clearAllocatedFields(){

    // field3DGraphicsData.clear();
    // field3DConcentration.clear();
   std::map<std::string,floatField3D_t*>::iterator mitr;
   
    for( mitr = floatField3DNameMap.begin(); mitr != floatField3DNameMap.end() ; ++mitr){
       if(mitr->second)
          delete mitr->second;
    }
   
    floatField3DNameMap.clear();

    
    for (map<std::string, vectorFieldCellLevel_t * >::iterator vitr=vectorFieldCellLevelNameMap.begin() ; vitr!=vectorFieldCellLevelNameMap.end() ; ++vitr){
       if(vitr->second)
          delete vitr->second;
    }

    vectorFieldCellLevelNameMap.clear();

    for (map<std::string, scalarFieldCellLevel_t * >::iterator vitr=scalarFieldCellLevelNameMap.begin() ; vitr!=scalarFieldCellLevelNameMap.end() ; ++vitr){
       if(vitr->second)
          delete vitr->second;
    }

    scalarFieldCellLevelNameMap.clear();


   // for (map<std::string,vectorFloatField3D_t*>::iterator vitr=vectorFloatField3DNameMap.begin() ; vitr!=vectorFloatField3DNameMap.end(); ++vitr){
      // if(vitr->second)
         // delete vitr->second;
      
   // }
   // vectorFloatField3DNameMap.clear();

   // for(map<std::string,vectorCellFloatField3D_t*>::iterator vitr=vectorCellFloatField3DNameMap.begin() ; vitr!=vectorCellFloatField3DNameMap.end();++vitr){
      // if(vitr->second)
         // delete vitr->second;
      
   // }
   // vectorCellFloatField3DNameMap.clear();
   
   // //clearing plotNamePLotTypeMap
   // plotNamePlotTypeMap.clear();
 }



FieldStorage::~FieldStorage(){

   clearAllocatedFields();

}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void FieldStorage::insertPlotNamePlotTypePair(const std::string & _plotName , const std::string & _plotType){

// plotNamePlotTypeMap.insert(make_pair(_plotName,_plotType));
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// std::string FieldStorage::checkPlotType(const std::string & _plotName){

   // plotNamePlotTypeMap_t::iterator mitr;
   // mitr=plotNamePlotTypeMap.find(_plotName);

   // if(mitr != plotNamePlotTypeMap.end()){
      // return mitr->second;
   // }else{
      // return string("unknown");
   // }
   

// }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::allocateCellField(Dim3D _dim){
	fieldDim=_dim;
   
   field3DGraphicsData.assign(fieldDim.x,vector<vector<CellGraphicsData> >(fieldDim.y,vector<CellGraphicsData>(fieldDim.z)));


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::floatField3D_t* FieldStorage::createFloatFieldPy(Dim3D _dim,std::string _fieldName){
    floatField3D_t * floatFieldPtr = new floatField3D_t ;
   
    floatFieldPtr->assign(_dim.x,vector<vector<float> >(_dim.y,vector<float>(_dim.z)));
    floatField3DNameMap.insert(std::make_pair(_fieldName,floatFieldPtr));
	 return floatFieldPtr ;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::fillScalarValue(FieldStorage::floatField3D_t * _field, int _x, int _y, int _z, float _value){
	(*_field)[_x][_y][_z]=_value;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::clearScalarField(Dim3D _dim, FieldStorage::floatField3D_t * _fieldPtr){
	 
	 for (int x=0;x<_dim.x;++x)
		for (int y=0;y<_dim.y;++y)
			for (int z=0;z<_dim.z;++z){
				(*_fieldPtr)[x][y][z]=0.0;
			}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldStorage::getScalarFieldNameVector(){
	vector<string> fieldNameVec;
	floatField3DNameMapItr_t mitr;
	for (mitr=floatField3DNameMap.begin()  ; mitr !=floatField3DNameMap.end() ; ++mitr){
		fieldNameVec.push_back(mitr->first);	
	}
	return fieldNameVec;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::floatField3D_t * FieldStorage::getScalarFieldByName(std::string _fieldName){

	floatField3DNameMapItr_t mitr=floatField3DNameMap.find(_fieldName);
	if(mitr!=floatField3DNameMap.end())
		return mitr->second;
	else
		return 0;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::vectorFieldCellLevel_t * FieldStorage::createVectorFieldCellLevelPy(std::string _fieldName){
    vectorFieldCellLevel_t * fieldPtr = new vectorFieldCellLevel_t;
    vectorFieldCellLevelNameMap.insert(make_pair(_fieldName,fieldPtr));
	 return fieldPtr ;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::insertVectorIntoVectorCellLevelField(FieldStorage::vectorFieldCellLevel_t * _field,CompuCell3D::CellG* _cell, float _x, float _y, float _z){
	_field->insert(std::make_pair(_cell,Coordinates3D<float>(_x,_y,_z)));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::clearVectorCellLevelField(FieldStorage::vectorFieldCellLevel_t * _field){
	_field->clear();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldStorage::getVectorFieldCellLevelNameVector(){
	vector<string> fieldNameVec;
	vectorFieldCellLevelNameMapItr_t mitr;
	for (mitr=vectorFieldCellLevelNameMap.begin()  ; mitr !=vectorFieldCellLevelNameMap.end() ; ++mitr){
		fieldNameVec.push_back(mitr->first);	
	}
	return fieldNameVec;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::vectorFieldCellLevel_t * FieldStorage::getVectorFieldCellLevelFieldByName(std::string _fieldName){
	vectorFieldCellLevelNameMapItr_t mitr=vectorFieldCellLevelNameMap.find(_fieldName);
	if(mitr!=vectorFieldCellLevelNameMap.end())
		return mitr->second;
	else
		return 0;
}









////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::vectorField3D_t * FieldStorage::createVectorFieldPy(Dim3D _dim , std::string _fieldName){
    vectorField3D_t * fieldPtr = new vectorField3D_t;
	 fieldPtr->assign(_dim.x,vector<vector<Coordinates3D<float> > >(_dim.y,vector<Coordinates3D<float> >(_dim.z)));
	 for (int x=0;x<_dim.x;++x)
		for (int y=0;y<_dim.y;++y)
			for (int z=0;z<_dim.z;++z){
				(*fieldPtr)[x][y][z]=Coordinates3D<float>(0.,0.,0.);
			}
    vectorFieldNameMap.insert(make_pair(_fieldName,fieldPtr));
	 return fieldPtr ;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::insertVectorIntoVectorField(FieldStorage::vectorField3D_t * _field,int _xPos, int _yPos, int _zPos, float _x, float _y, float _z){
	(*_field)[_xPos][_yPos][_zPos]=Coordinates3D<float>(_x,_y,_z);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::clearVectorField(Dim3D _dim, FieldStorage::vectorField3D_t * _fieldPtr){
	 
	 for (int x=0;x<_dim.x;++x)
		for (int y=0;y<_dim.y;++y)
			for (int z=0;z<_dim.z;++z){
				(*_fieldPtr)[x][y][z]=Coordinates3D<float>(0.,0.,0.);
			}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldStorage::getVectorFieldNameVector(){
	vector<string> fieldNameVec;
	vectorFieldNameMapItr_t mitr;
	for (mitr=vectorFieldNameMap.begin()  ; mitr !=vectorFieldNameMap.end() ; ++mitr){
		fieldNameVec.push_back(mitr->first);	
	}
	return fieldNameVec;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::vectorField3D_t * FieldStorage::getVectorFieldFieldByName(std::string _fieldName){
	vectorFieldNameMapItr_t mitr=vectorFieldNameMap.find(_fieldName);
	if(mitr!=vectorFieldNameMap.end())
		return mitr->second;
	else
		return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::scalarFieldCellLevel_t * FieldStorage::createScalarFieldCellLevelPy(std::string _fieldName){
    scalarFieldCellLevel_t * fieldPtr = new scalarFieldCellLevel_t;
    scalarFieldCellLevelNameMap.insert(make_pair(_fieldName,fieldPtr));
	 return fieldPtr ;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::insertScalarIntoScalarCellLevelField(FieldStorage::scalarFieldCellLevel_t * _field,CompuCell3D::CellG* _cell, float _x){
	_field->insert(std::make_pair(_cell,_x));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::clearScalarCellLevelField(FieldStorage::scalarFieldCellLevel_t * _field){
	_field->clear();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldStorage::getScalarFieldCellLevelNameVector(){
	vector<string> fieldNameVec;
	scalarFieldCellLevelNameMapItr_t mitr;
	for (mitr=scalarFieldCellLevelNameMap.begin()  ; mitr !=scalarFieldCellLevelNameMap.end() ; ++mitr){
		fieldNameVec.push_back(mitr->first);	
	}
	return fieldNameVec;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::scalarFieldCellLevel_t * FieldStorage::getScalarFieldCellLevelFieldByName(std::string _fieldName){
	scalarFieldCellLevelNameMapItr_t mitr=scalarFieldCellLevelNameMap.find(_fieldName);
	if(mitr!=scalarFieldCellLevelNameMap.end())
		return mitr->second;
	else
		return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void FieldStorage::allocateFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN , std::string _name){

   // floatField3D_t * floatFieldPtr = new floatField3D_t ;
   
   // floatFieldPtr->assign(_sizeL,vector<vector<float> >(_sizeM,vector<float>(_sizeN)));
   // floatField3DNameMap.insert(std::make_pair(_name,floatFieldPtr));

// }
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void FieldStorage::allocateVectorFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN , std::string _name){

   // vectorFloatField3D_t * fieldPtr = new vectorFloatField3D_t ;
   
   // fieldPtr->assign(_sizeL,vector<vector<Coordinates3D<float> > >(_sizeM,vector<Coordinates3D<float> >(_sizeN)));
   // vectorFloatField3DNameMap.insert(std::make_pair(_name,fieldPtr));

// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void FieldStorage::allocateVectorCellFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN , std::string _name){

   // vectorCellFloatField3D_t * fieldPtr = new vectorCellFloatField3D_t ;
   
   // fieldPtr->assign(_sizeL,vector<vector<pair<Coordinates3D<float>,CompuCell3D::CellG*> > >
   // (_sizeM,vector<pair<Coordinates3D<float>,CompuCell3D::CellG*> >(_sizeN)));
   // vectorCellFloatField3DNameMap.insert(std::make_pair(_name,fieldPtr));
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void FieldStorage::allocateVectorFieldCellLevel(std::string _name){

   // vectorFieldCellLevel_t * fieldPtr = new vectorFieldCellLevel_t;
   // vectorFieldCellLevelNameMap.insert(make_pair(_name,fieldPtr));

// }