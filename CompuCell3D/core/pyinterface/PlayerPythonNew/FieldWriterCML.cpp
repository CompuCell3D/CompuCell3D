/**
 * @file FieldWriterCML.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines FieldWriterCML, a field writer for simulation-independent data throughput with VTK
 * @date 2022-04-18
 * 
 */

#include "FieldWriterCML.h"
#include "FieldStorage.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/plugins/FocalPointPlasticity/FocalPointPlasticityPlugin.h>

#include <vtkDataArray.h>
#include <vtkCharArray.h>
#include <vtkDoubleArray.h>
#include <vtkLongArray.h>
#include <vtkPointData.h>


#define FIELDWRITERCML_CHECKADD { if(!sim || !fsPtr || !latticeData) return false; }

using namespace CompuCell3D;

const std::string FieldWriterCML::CellTypeName = "CellType";
const std::string FieldWriterCML::CellIdName = "CellId";
const std::string FieldWriterCML::ClusterIdName = "ClusterId";
const std::string FieldWriterCML::LinksName = "Links";
const std::string FieldWriterCML::LinksInternalName = "LinksInternal";
const std::string FieldWriterCML::AnchorsName = "Anchors";

FieldWriterCML::FieldWriterCML() : 
    sim(NULL), 
    fsPtr(NULL), 
    latticeData(NULL)
{}

FieldWriterCML::~FieldWriterCML() {
    this->clear();
    if(sim != NULL) 
        sim = NULL;
    if(fsPtr != NULL) 
        fsPtr = NULL;
    if(latticeData != NULL) {
        latticeData->Delete();
    }
}

void FieldWriterCML::init(Simulator *_sim) {
    sim = _sim;
    latticeData = vtkStructuredPoints::New();
    Dim3D fieldDim = sim->getPotts()->getCellFieldG()->getDim();
    latticeData->SetDimensions(fieldDim.x, fieldDim.y, fieldDim.z);
}

void FieldWriterCML::clear() {
    if(latticeData) 
        for(auto &s : arrayNameVec) 
            latticeData->GetPointData()->RemoveArray(s.c_str());
    
    arrayNameVec.clear();
    arrayTypeVec.clear();
}

std::string FieldWriterCML::getFieldName(const unsigned int &i) const {
    if(i >= numFields()) {
        // todo: implement error on rebase
        return "";
    }
    return arrayNameVec[i];
}

FieldTypeCML FieldWriterCML::getFieldType(const unsigned int &i) const {
    if(i >= numFields()) {
        // todo: implement error on rebase
        return (FieldTypeCML)0;
    }
    return arrayTypeVec[i];
}

vtk_obj_addr_int_t FieldWriterCML::getArrayAddr(const unsigned int &i) const {
    if(i >= numFields()) {
        // todo: implement error on rebase
        return 0;
    }
    return (vtk_obj_addr_int_t)latticeData->GetPointData()->GetArray(i);
}

vtk_obj_addr_int_t FieldWriterCML::getArrayAddr(const std::string &name) const {
    for(unsigned int i = 0; i < numFields(); i++) 
        if(arrayNameVec[i] == name) 
            return (vtk_obj_addr_int_t)latticeData->GetPointData()->GetArray(i);
    return 0;
}

Dim3D FieldWriterCML::getFieldDim() const {
    if(!sim) 
        return Dim3D();
    return sim->getPotts()->getCellFieldG()->getDim();
}

bool FieldWriterCML::addCellFieldForOutput() {
    FIELDWRITERCML_CHECKADD
    
    auto cellFieldG = sim->getPotts()->getCellFieldG();
    auto fieldDim = cellFieldG->getDim();

    vtkCharArray *typeArray = vtkCharArray::New();
    typeArray->SetName(FieldWriterCML::CellTypeName.c_str());
    arrayNameVec.push_back(FieldWriterCML::CellTypeName);
    arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_CellField);

    vtkLongArray *idArray = vtkLongArray::New();
    idArray->SetName(FieldWriterCML::CellIdName.c_str());
    arrayNameVec.push_back(FieldWriterCML::CellIdName);
    arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_CellField);

    vtkLongArray *clusterIdArray = vtkLongArray::New();
    clusterIdArray->SetName(FieldWriterCML::ClusterIdName.c_str());
    arrayNameVec.push_back(FieldWriterCML::ClusterIdName);
    arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_CellField);

    long numberOfValues = fieldDim.x * fieldDim.y * fieldDim.z;

    typeArray->SetNumberOfValues(numberOfValues);
    idArray->SetNumberOfValues(numberOfValues);
    clusterIdArray->SetNumberOfValues(numberOfValues);

    long offset = 0;
    CellG *cell;
    Point3D pt;

    for(pt.z = 0; pt.z < fieldDim.z; pt.z++) 
        for(pt.y = 0; pt.y < fieldDim.y; pt.y++) 
            for(pt.x = 0; pt.x < fieldDim.x; pt.x++) {
                cell = cellFieldG->get(pt);
                if(cell) {
                    typeArray->SetValue(offset, cell->type);
                    idArray->SetValue(offset, cell->id);
                    clusterIdArray->SetValue(offset, cell->clusterId);
                }
                else {
                    typeArray->SetValue(offset, 0);
                    idArray->SetValue(offset, 0);
                    clusterIdArray->SetValue(offset, 0);
                }
                ++offset;
            }

    latticeData->GetPointData()->AddArray(typeArray);
    latticeData->GetPointData()->AddArray(idArray);
    latticeData->GetPointData()->AddArray(clusterIdArray);

    typeArray->Delete();
    idArray->Delete();
    clusterIdArray->Delete();

    FocalPointPlasticityPlugin *fppPlugin;
    if(sim->pluginManager.isLoaded("FocalPointPlasticity")) {
        fppPlugin = (FocalPointPlasticityPlugin*)sim->pluginManager.get("FocalPointPlasticity");

        auto listLinks = fppPlugin->getLinkInventory()->getLinkList();
        auto listLinksInternal = fppPlugin->getInternalLinkInventory()->getLinkList();
        auto listAnchors = fppPlugin->getAnchorInventory()->getLinkList();

        vtkLongArray *linksArray = vtkLongArray::New();
        linksArray->SetName(FieldWriterCML::LinksName.c_str());
        arrayNameVec.push_back(FieldWriterCML::LinksName);
        arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_Links);
        
        linksArray->SetNumberOfComponents(2);
        linksArray->SetNumberOfTuples(listLinks.size());

        for(int i = 0; i < listLinks.size(); i++) {
            FocalPointPlasticityLink *link = listLinks[i];
            linksArray->SetTuple2(i, link->getId0(), link->getId1());
        }

        latticeData->GetPointData()->AddArray(linksArray);
        linksArray->Delete();

        vtkLongArray *linksInternalArray = vtkLongArray::New();
        linksInternalArray->SetName(FieldWriterCML::LinksInternalName.c_str());
        arrayNameVec.push_back(FieldWriterCML::LinksInternalName);
        arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_LinksInternal);

        linksInternalArray->SetNumberOfComponents(2);
        linksInternalArray->SetNumberOfTuples(listLinksInternal.size());

        for(int i = 0; i < listLinksInternal.size(); i++) {
            FocalPointPlasticityInternalLink *link = listLinksInternal[i];
            linksInternalArray->SetTuple2(i, link->getId0(), link->getId1());
        }

        latticeData->GetPointData()->AddArray(linksInternalArray);
        linksInternalArray->Delete();

        vtkDoubleArray *anchorsArray = vtkDoubleArray::New();
        anchorsArray->SetName(FieldWriterCML::AnchorsName.c_str());
        arrayNameVec.push_back(FieldWriterCML::AnchorsName);
        arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_Anchors);

        anchorsArray->SetNumberOfComponents(4);
        anchorsArray->SetNumberOfTuples(listAnchors.size());

        for(int i = 0; i < listAnchors.size(); i++) {
            FocalPointPlasticityAnchor *link = listAnchors[i];
            auto pt = link->getAnchorPoint();
            anchorsArray->SetTuple4(i, link->getId0(), pt[0], pt[1], pt[2]);
        }

        latticeData->GetPointData()->AddArray(anchorsArray);
        anchorsArray->Delete();
    }
    
    return true;
}

bool FieldWriterCML::addConFieldForOutput(const std::string &_conFieldName) {
    FIELDWRITERCML_CHECKADD
    
    auto cellFieldG = sim->getPotts()->getCellFieldG();
    auto fieldDim = cellFieldG->getDim();

    std::map<std::string, Field3D<float>*> &fieldMap = sim->getConcentrationFieldNameMap();
	std::map<std::string, Field3D<float>*>::iterator mitr = fieldMap.find(_conFieldName);
    if(mitr == fieldMap.end()) 
        return false;
	auto conFieldPtr = mitr->second;

    vtkDoubleArray *conArray = vtkDoubleArray::New();
    conArray->SetName(_conFieldName.c_str());
    arrayNameVec.push_back(_conFieldName);
    arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_ConField);

    conArray->SetNumberOfValues(fieldDim.x * fieldDim.y * fieldDim.z);
    long offset = 0;
    Point3D pt;
    
    for(pt.z = 0; pt.z < fieldDim.z; pt.z++) 
        for(pt.y = 0; pt.y < fieldDim.y; pt.y++) 
            for(pt.x = 0; pt.x < fieldDim.x; pt.x++) {
                conArray->SetValue(offset, conFieldPtr->get(pt));
                ++offset;
            }

    latticeData->GetPointData()->AddArray(conArray);

    conArray->Delete();

    return true;
}

bool FieldWriterCML::addScalarFieldForOutput(const std::string &_scalarFieldName) {
    FIELDWRITERCML_CHECKADD
    
    auto cellFieldG = sim->getPotts()->getCellFieldG();
    auto fieldDim = cellFieldG->getDim();

    FieldStorage::floatField3D_t *conFieldPtr = fsPtr->getScalarFieldByName(_scalarFieldName); 
	if(!conFieldPtr)
		return false;

	vtkDoubleArray *conArray = vtkDoubleArray::New();
	conArray->SetName(_scalarFieldName.c_str());
	arrayNameVec.push_back(_scalarFieldName);	
    arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_ScalarField);

	conArray->SetNumberOfValues(fieldDim.x * fieldDim.y * fieldDim.z);
	long offset = 0;

	for(short z = 0; z < fieldDim.z; ++z)	
		for(short y = 0; y < fieldDim.y; ++y)
			for(short x = 0; x < fieldDim.x; ++x){
				conArray->SetValue(offset, (*conFieldPtr)[x][y][z]);
				++offset;
			}
	
    latticeData->GetPointData()->AddArray(conArray);

	conArray->Delete();

	return true;
}

bool FieldWriterCML::addScalarFieldCellLevelForOutput(const std::string &_scalarFieldCellLevelName) {
    FIELDWRITERCML_CHECKADD
    
    auto cellFieldG = sim->getPotts()->getCellFieldG();
    auto fieldDim = cellFieldG->getDim();

    FieldStorage::scalarFieldCellLevel_t *conFieldPtr = fsPtr->getScalarFieldCellLevelFieldByName(_scalarFieldCellLevelName);
    if(!conFieldPtr) 
        return false;

    vtkDoubleArray *conArray = vtkDoubleArray::New();
	conArray->SetName(_scalarFieldCellLevelName.c_str());
	arrayNameVec.push_back(_scalarFieldCellLevelName);
    arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_ScalarFieldCellLevel);

    conArray->SetNumberOfValues(fieldDim.x * fieldDim.y * fieldDim.z);
    long offset = 0;
    CellG *cell;
    Point3D pt;
    FieldStorage::scalarFieldCellLevelItr_t mitr;
    float con;

    for(pt.z = 0; pt.z < fieldDim.z; ++pt.z)	
		for(pt.y = 0; pt.y < fieldDim.y; ++pt.y)
			for(pt.x = 0; pt.x < fieldDim.x; ++pt.x) {
                cell = cellFieldG->get(pt);
                if(cell) {
                    mitr = conFieldPtr->find(cell);
                    if(mitr == conFieldPtr->end()) 
                        con = 0.0;
                    else 
                        con = mitr->second;
                } 
                else {
                    con = 0.0;
                }
                conArray->SetValue(offset, con);
                ++offset;
            }

    latticeData->GetPointData()->AddArray(conArray);

    conArray->Delete();

    return true;
}

bool FieldWriterCML::addVectorFieldForOutput(const std::string &_vectorFieldName) {
    FIELDWRITERCML_CHECKADD

	auto cellFieldG = sim->getPotts()->getCellFieldG();
	auto fieldDim = cellFieldG->getDim();

	FieldStorage::vectorField3D_t *vecFieldPtr=fsPtr->getVectorFieldFieldByName(_vectorFieldName);

	if(!vecFieldPtr)
		return false;

	vtkDoubleArray *vecArray = vtkDoubleArray::New();
	vecArray->SetNumberOfComponents(3);
	vecArray->SetName(_vectorFieldName.c_str());
	arrayNameVec.push_back(_vectorFieldName);
    arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_VectorField);

	vecArray->SetNumberOfTuples(fieldDim.x * fieldDim.y * fieldDim.z);
	long offset = 0;

	for(short z = 0 ; z < fieldDim.z; ++z)	
		for(short y = 0 ; y < fieldDim.y; ++y)
			for(short x = 0 ; x < fieldDim.x; ++x) { 
                float *v = &(*vecFieldPtr)[x][y][z][0];
				vecArray->SetTuple3(offset, v[0], v[1], v[2]);
				++offset;
			}

	latticeData->GetPointData()->AddArray(vecArray);

	vecArray->Delete();

	return true;
}

bool FieldWriterCML::addVectorFieldCellLevelForOutput(const std::string &_vectorFieldCellLevelName) {
    FIELDWRITERCML_CHECKADD

	auto cellFieldG = sim->getPotts()->getCellFieldG();
	auto fieldDim = cellFieldG->getDim();

	FieldStorage::vectorFieldCellLevel_t *vecFieldPtr = fsPtr->getVectorFieldCellLevelFieldByName(_vectorFieldCellLevelName);
	if(!vecFieldPtr)
		return false;

	vtkDoubleArray *vecArray = vtkDoubleArray::New();
	vecArray->SetNumberOfComponents(3);
	vecArray->SetName(_vectorFieldCellLevelName.c_str());
	arrayNameVec.push_back(_vectorFieldCellLevelName);	
    arrayTypeVec.push_back(FieldTypeCML::FieldTypeCML_VectorFieldCellLevel);

	vecArray->SetNumberOfTuples(fieldDim.x * fieldDim.y * fieldDim.z);
	long offset=0;
	Point3D pt;
	
	Coordinates3D<float> vecTmp;
	FieldStorage::vectorFieldCellLevelItr_t mitr;

	CellG * cell;

	for(pt.z = 0; pt.z < fieldDim.z; ++pt.z)	
		for(pt.y = 0; pt.y < fieldDim.y; ++pt.y)
			for(pt.x = 0; pt.x < fieldDim.x; ++pt.x) {
				cell=cellFieldG->get(pt);
				if(cell) {
					mitr = vecFieldPtr->find(cell);
					if(mitr!=vecFieldPtr->end())
						vecTmp = mitr->second;
					else 
                        vecTmp = Coordinates3D<float>();
				} 
                else 
                    vecTmp = Coordinates3D<float>();

                vecArray->SetTuple3(offset, vecTmp.x, vecTmp.y, vecTmp.z);
				++offset;
			}

    latticeData->GetPointData()->AddArray(vecArray);
	
    vecArray->Delete();

	return true;
}

bool FieldWriterCML::addFieldForOutput(const std::string &_fieldName) {
    FIELDWRITERCML_CHECKADD
    
    for(auto &s : sim->getConcentrationFieldNameVector()) 
        if(_fieldName == s) 
            return addConFieldForOutput(_fieldName);
    for(auto &s : fsPtr->getScalarFieldNameVector()) 
        if(_fieldName == s) 
            return addScalarFieldForOutput(_fieldName);
    for(auto &s : fsPtr->getScalarFieldCellLevelNameVector()) 
        if(_fieldName == s) 
            return addScalarFieldCellLevelForOutput(_fieldName);
    for(auto &s : fsPtr->getVectorFieldNameVector()) 
        if(_fieldName == s) 
            return addVectorFieldForOutput(_fieldName);
    for(auto &s : fsPtr->getVectorFieldCellLevelNameVector()) 
        if(_fieldName == s) 
            return addVectorFieldCellLevelForOutput(_fieldName);
    
    return false;
}
