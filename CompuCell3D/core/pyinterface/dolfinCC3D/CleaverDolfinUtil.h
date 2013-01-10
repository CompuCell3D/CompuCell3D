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

void dolfinMeshInfo(void *_obj);  
void simulateCleaverMesh(void *_cellField,std::vector<unsigned char> _includeTypesVec);

void buildCellFieldDolfinMeshUsingCleaver(void *_cellField,void *_dolfinMesh ,std::vector<unsigned char> _includeTypesVec,bool _verbose=true);


#endif