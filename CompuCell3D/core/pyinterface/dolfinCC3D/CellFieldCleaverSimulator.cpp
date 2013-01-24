
#include <iostream>

#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Potts3D/Cell.h>

#include <Cleaver/Cleaver.h>
#include <Cleaver/InverseField.h>
#include <Cleaver/FloatField.h>

#include "CellFieldCleaverSimulator.h"





using namespace Cleaver;
using namespace std;  
using namespace CompuCell3D;

CellFieldCleaverSimulatorNew::CellFieldCleaverSimulatorNew() : 
m_bounds(vec3::zero, vec3(1,1,1)),paddingDim(2,2,2),cellField(0)
{
    // no allocation
    minValue=1000000000.0;
    maxValue=-1000000000.0;        
}

CellFieldCleaverSimulatorNew::~CellFieldCleaverSimulatorNew()
{
    // no memory cleanup
}

BoundingBox CellFieldCleaverSimulatorNew::bounds() const
{
    return m_bounds;
}

void CellFieldCleaverSimulatorNew::setFieldDim(Dim3D & _dim){
    fieldDim=_dim;    
    m_bounds.size=vec3(fieldDim.x,fieldDim.y,fieldDim.z);
}



float CellFieldCleaverSimulatorNew::valueAt(float x, float y, float z) const
{


    int dim_x = m_bounds.size.x;
    int dim_y = m_bounds.size.y;
    int dim_z = m_bounds.size.z;

    // Current Cleaver Limitation - Can't have material transitions on the boundary.
    // Will fix ASAP, but for now pad data with constant boundary.
    if(x < paddingDim.x || y < paddingDim.y || z < paddingDim.z || x > (dim_x - paddingDim.x) || y > (dim_y - paddingDim.y) || z > (dim_z - paddingDim.z))
    {
        return -11.0;

    }

    CellG * cell=cellField->get(Point3D(x,y,z));

    


    if (! cell){
        return -9.0;
    }else if (includeCellTypesSet.find(cell->type)!=includeCellTypesSet.end()){ //first check cell type
        return 2.0+cell->type;
	
    } else if (includeCellIdsSet.find(cell->id)!=includeCellIdsSet.end()){ //then cell id - this way we can have 'additive' specification of what to mesh
									   // and order is important here	
        return 2.0+cell->type;	
    }else {
        return -9.0;
    }

    //if (! cell){
    //  return -9.0;
    //}else if (includeCellTypesSet.find(cell->type)!=end_sitr){
    //  return 2.0+cell->type;
    //} else {
    //  return -9.0;
    //}

    //if (! cell){
    //  return -9.0;
    //}else if (cell->type==1){
    //  return 2.0+cell->type;
    //} else {
    //  return -9.0;
    //}
}


