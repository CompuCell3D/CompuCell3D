/**
 * @file FieldWriterCML.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines FieldWriterCML, a field writer for simulation-independent data throughput with VTK
 * @date 2022-04-18
 * 
 */
#ifndef FIELDWRITERCML_H
#define FIELDWRITERCML_H

#include "FieldExtractorDLLSpecifier.h"

#include "FieldStorage.h"
#include "FieldExtractorTypes.h"

#include <vtkStructuredPoints.h>

#include <vector>
#include <string>
#include <functional>

namespace CompuCell3D {

    class Simulator;
    class FieldStreamer;

    enum FIELDEXTRACTOR_EXPORT FieldTypeCML : unsigned int {
        FieldTypeCML_CellField,
        FieldTypeCML_ConField,
        FieldTypeCML_ScalarField, 
        FieldTypeCML_ScalarFieldCellLevel, 
        FieldTypeCML_VectorField, 
        FieldTypeCML_VectorFieldCellLevel,
        FieldTypeCML_Links,
        FieldTypeCML_LinksInternal,
        FieldTypeCML_Anchors
    };

    class FIELDEXTRACTOR_EXPORT FieldWriterCML {

		Simulator *sim;
        FieldStorage *fsPtr;
        vtkStructuredPoints *latticeData;
        std::vector<std::string> arrayNameVec;
        std::vector<FieldTypeCML> arrayTypeVec;

    public:

        static const std::string CellTypeName;
        static const std::string CellIdName;
        static const std::string ClusterIdName;
        static const std::string LinksName;
        static const std::string LinksInternalName;
        static const std::string AnchorsName;

		FieldWriterCML();
		~FieldWriterCML();
        
        void init(Simulator *_sim);
        void clear();
        void setFieldStorage(FieldStorage *_fsPtr) { fsPtr = _fsPtr; }
        FieldStorage *getFieldStorage() { return fsPtr; }

        size_t numFields() const { return arrayNameVec.size(); }
        std::string getFieldName(const unsigned int &i) const;
        FieldTypeCML getFieldType(const unsigned int &i) const;
        vtk_obj_addr_int_t getArrayAddr(const unsigned int &i) const;
        vtk_obj_addr_int_t getArrayAddr(const std::string &name) const;
        Dim3D getFieldDim() const;

        bool addCellFieldForOutput();
        bool addConFieldForOutput(const std::string &_conFieldName);
        bool addScalarFieldForOutput(const std::string &_scalarFieldName);
        bool addScalarFieldCellLevelForOutput(const std::string &_scalarFieldCellLevelName);
        bool addVectorFieldForOutput(const std::string &_vectorFieldName);
        bool addVectorFieldCellLevelForOutput(const std::string &_vectorFieldCellLevelName);
        bool addFieldForOutput(const std::string &_fieldName);

        friend FieldStreamer;
    
    };

};

#endif // FIELDWRITERCML_H