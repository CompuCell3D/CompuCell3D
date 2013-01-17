#ifndef CLEAVERDOLFINUTIL_H
#define CLEAVERDOLFINUTIL_H

#include <vector>


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

void buildCellFieldDolfinMeshUsingCleaver(void *_cellField,void *_dolfinMesh ,std::vector<unsigned char> _includeTypesVec,bool _verbose=true);

// void extractSolutionValuesAtLatticePoints(void *_cellField, void *_dolfinSolutionFunction);
// // // void extractSolutionValuesAtLatticePoints(void *_cellField, dolfin::Function *_dolfinSolutionFunction);
// // // void extractSolutionValuesAtLatticePoints(void *_cellField, dolfin::Function *_dolfinSolutionFunction,void *_dptr);
// void extractSolutionValuesAtLatticePoints(void *_cellField, void *_dptr);
void extractSolutionValuesAtLatticePoints(void *_scalarField, void *_dptr);
#endif