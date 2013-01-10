#ifndef CELLFIELDCLEAVERSIMULATOR_H
#define CELLFIELDCLEAVERSIMULATOR_H

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>

//STL containers
#include <vector>
#include <list>
#include <set>
#include <map>


#include <Cleaver/ScalarField.h>
#include <Cleaver/BoundingBox.h>

#include "DolfinCC3DDLLSpecifier.h"

namespace CompuCell3D {
    
    template <class T> class Field3D;
    template <class T> class WatchableField3D;

    class Potts3D;
    class Automaton;
    class BoundaryStrategy;
    class CellInventory;
    class CellG;
    class Dim3D;

  
    class DOLFINCC3D_EXPORT CellFieldCleaverSimulatorNew : public Cleaver::ScalarField
    {
    public:
        CellFieldCleaverSimulatorNew();
        ~CellFieldCleaverSimulatorNew();
    
        virtual float valueAt(float x, float y, float z) const;
        virtual Cleaver::BoundingBox bounds() const;
    
        mutable float minValue,maxValue;
        void setFieldDim(Dim3D & _dim);
        void setCellFieldPtr(WatchableField3D<CellG *> * _cellField){cellField=_cellField;}
        void setIncludeCellTypesSet(std::set<unsigned char> & _cellTypeSet){
            includeCellTypesSet=_cellTypeSet;
            end_sitr=includeCellTypesSet.end();
        }
        
        void addIncludeCellTypesSet(unsigned char  _type){            
	    includeCellTypesSet.insert(_type);
            end_sitr=includeCellTypesSet.end();
        }
        
    private:
        Cleaver::BoundingBox m_bounds;
        Dim3D fieldDim;
        Dim3D paddingDim;
        WatchableField3D<CellG *> * cellField;
        std::set<unsigned char> includeCellTypesSet;
        std::set<unsigned char>::iterator end_sitr;

            
    };
    
  
    
    
};


#endif