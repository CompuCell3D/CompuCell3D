#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;


#include <iostream>

using namespace std;

#include "CleaverMeshDumper.h"


#include <Cleaver/Cleaver.h>
#include <Cleaver/InverseField.h>
#include <Cleaver/FloatField.h>
#include <Logger/CC3DLogger.h>

using namespace Cleaver;


CellFieldCleaverSimulator::CellFieldCleaverSimulator() :
        m_bounds(vec3::zero, vec3(1, 1, 1)), paddingDim(2, 2, 2), cellField(0) {
    // no allocation
    minValue = 1000000000.0;
    maxValue = -1000000000.0;
}

CellFieldCleaverSimulator::~CellFieldCleaverSimulator() {
    // no memory cleanup
}

BoundingBox CellFieldCleaverSimulator::bounds() const {
    return m_bounds;
}

void CellFieldCleaverSimulator::setFieldDim(Dim3D _dim) {
    fieldDim = _dim;
    m_bounds.size = vec3(fieldDim.x, fieldDim.y, fieldDim.z);
}


float CellFieldCleaverSimulator::valueAt(float x, float y, float z) const {


    int dim_x = m_bounds.size.x;
    int dim_y = m_bounds.size.y;
    int dim_z = m_bounds.size.z;

    // Current Cleaver Limitation - Can't have material transitions on the boundary.
    // Will fix ASAP, but for now pad data with constant boundary.
    if (x < paddingDim.x || y < paddingDim.y || z < paddingDim.z || x > (dim_x - paddingDim.x) ||
        y > (dim_y - paddingDim.y) || z > (dim_z - paddingDim.z)) {
        return -11.0;

    }

    CellG *cell = cellField->get(Point3D(x, y, z));


    if (!cell) {
        return -9.0;
    } else if (includeCellTypesSet.find(cell->type) != includeCellTypesSet.end()) {
        return 2.0 + cell->type;
    } else {
        return -9.0;
    }

    //if (! cell){
    //	return -9.0;
    //}else if (includeCellTypesSet.find(cell->type)!=end_sitr){
    //	return 2.0+cell->type;
    //} else {
    //	return -9.0;
    //}

    //if (! cell){
    //	return -9.0;
    //}else if (cell->type==1){
    //	return 2.0+cell->type;
    //} else {
    //	return -9.0;
    //}
}


CleaverMeshDumper::CleaverMeshDumper() :
        cellFieldG(0), sim(0), potts(0),
        xmlData(0), boundaryStrategy(0), automaton(0), cellInventoryPtr(0) {
    meshOutputFormat = "tetgen";
    outputMeshSurface = false;
    verbose = false;

}

CleaverMeshDumper::~CleaverMeshDumper() {
}


void CleaverMeshDumper::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;

    potts = simulator->getPotts();
    cellInventoryPtr = &potts->getCellInventory();
    sim = simulator;
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();


    simulator->registerSteerableObject(this);

    update(_xmlData, true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CleaverMeshDumper::extraInit(Simulator *simulator) {
    //PUT YOUR CODE HERE
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CleaverMeshDumper::start() {

    //PUT YOUR CODE HERE

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CleaverMeshDumper::simulateCleaverMesh() {
    CellFieldCleaverSimulator cfcs;
    cfcs.setFieldDim(fieldDim);
    cfcs.setCellFieldPtr(cellFieldG);
    cfcs.setIncludeCellTypesSet(cellTypesSet);

    //bool verbose=true;
    //string outputFileName="cellfieldmesh";

    Cleaver::InverseField inverseField = Cleaver::InverseField(&cfcs);

    std::vector < Cleaver::ScalarField * > fields;

    fields.push_back(&cfcs);
    fields.push_back(&inverseField);

    Cleaver::Volume volume(fields);
    Cleaver::TetMesh *mesh = Cleaver::createMeshFromVolume(volume, verbose);

    //mesh->writeNodeEle(outputFileName, verbose);
    CC3D_Log(LOG_DEBUG) << "outputFileName="<<outputFileName;
    CC3D_Log(LOG_DEBUG) << "verbose="<<verbose;


    if (meshOutputFormat == "tetgen")
        mesh->writeNodeEle(outputFileName, verbose);
    else if (meshOutputFormat == "scirun")
        mesh->writePtsEle(outputFileName, verbose);
    else if (meshOutputFormat == "matlab")
        mesh->writeMatlab(outputFileName, verbose);


    //----------------------
    // Write Surface Files
    //----------------------
    if (outputMeshSurface) {
        mesh->constructFaces();
        mesh->writePly(outputFileName, verbose);
    }


    delete mesh;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CleaverMeshDumper::step(const unsigned int currentStep){
	if (! currentStep%10){
		simulateCleaverMesh();
	}
	//   //REPLACE SAMPLE CODE BELOW WITH YOUR OWN
	//CellInventory::cellInventoryIterator cInvItr;
	//CellG * cell;
	//   CC3D_Log(LOG_DEBUG) << currentStep="<<currentStep;
	//for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )
	//{
	//	cell=cellInventoryPtr->getCell(cInvItr);
	// CC3D_Log(LOG_DEBUG) << "cell.id="<<cell->id<<" vol="<<cell->volume;
    //   }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CleaverMeshDumper::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");
    cellTypesSet.clear();

    std::vector <std::string> typeNames;


    CC3DXMLElement *outputXMLElem = _xmlData->getFirstElement("OutputSpecification");
    if (outputXMLElem) {
        outputFileName = outputXMLElem->getFirstElement("OutputFileNeme")->getText();
        if (outputXMLElem->getFirstElement("OutputMeshSurface")) {
            outputMeshSurface = true;
        }

        if (outputXMLElem->getFirstElement("Verbose")) {
            verbose = true;
            CC3D_Log(LOG_DEBUG) << "verbose="<<verbose;
        }


        if (outputXMLElem->getFirstElement("MeshOutputFormat")) {
            meshOutputFormat = outputXMLElem->getFirstElement("MeshOutputFormat")->getText();
        }

        if (outputXMLElem->getFirstElement("IncludeCellTypes")) {
            string celTypeStr = outputXMLElem->getFirstElement("IncludeCellTypes")->getText();
            parseStringIntoList(celTypeStr, typeNames, ",");
            for (int i = 0; i < typeNames.size(); ++i) {
                cellTypesSet.insert(automaton->getTypeId(typeNames[i]));

            }


            for (set<unsigned char>::iterator sitr = cellTypesSet.begin(); sitr != cellTypesSet.end(); ++sitr) {
                CC3D_Log(LOG_DEBUG) << "INCLUDIG CELL TYPE="<<(int)*sitr;
            }


        }
    }


    //boundaryStrategy has information aobut pixel neighbors
    boundaryStrategy = BoundaryStrategy::getInstance();

}

std::string CleaverMeshDumper::toString() {
    return "CleaverMeshDumper";
}

std::string CleaverMeshDumper::steerableName() {
    return toString();
}

