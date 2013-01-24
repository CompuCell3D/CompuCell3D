#ifndef CLEAVERDOLFINUTIL_H
#define CLEAVERDOLFINUTIL_H

#include <vector>
#include <CompuCell3D/Field3D/Dim3D.h>

#include "DolfinCC3DDLLSpecifier.h"

namespace CompuCell3D{
    class Dim3D;
    class CellG;
    template <class T> class Field3D;
    template <class T> class WatchableField3D;
    
}

namespace dolfin{
   class Function;
};

void dolfinMeshInfo(void *_obj);  
void simulateCleaverMesh(void *_cellField,std::vector<unsigned char> _includeTypesVec);

// void buildCellFieldDolfinMeshUsingCleaver(void *_cellField,void *_dolfinMesh ,std::vector<unsigned char>& _includeTypesVec, std::vector<long> & _includeIdsVec, bool _verbose=true);

//to enable automatic conversion from pythoh list to std vector we have to accept vector parameter either by value or a constant reference. 
void buildCellFieldDolfinMeshUsingCleaver(void *_cellField,void *_dolfinMesh , const std::vector<unsigned char>& _includeTypesVec=std::vector<unsigned char>(), const std::vector<long> & _includeIdsVec=std::vector<long>(), bool _verbose=true);

// void extractSolutionValuesAtLatticePoints(void *_cellField, void *_dolfinSolutionFunction);
// // // void extractSolutionValuesAtLatticePoints(void *_cellField, dolfin::Function *_dolfinSolutionFunction);
// // // void extractSolutionValuesAtLatticePoints(void *_cellField, dolfin::Function *_dolfinSolutionFunction,void *_dptr);
// void extractSolutionValuesAtLatticePoints(void *_cellField, void *_dptr);
void extractSolutionValuesAtLatticePoints(void *_scalarField, void *_dptr,CompuCell3D::Dim3D  boxMin=CompuCell3D::Dim3D(), CompuCell3D::Dim3D boxMax=CompuCell3D::Dim3D());
#endif