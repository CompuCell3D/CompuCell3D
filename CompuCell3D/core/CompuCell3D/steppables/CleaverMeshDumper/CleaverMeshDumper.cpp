



#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>

#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <PublicUtilities/StringUtils.h>
#include <algorithm>

//dolfin includes
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/MeshEditor.h>


using namespace CompuCell3D;


#include <iostream>
using namespace std;

#include "CleaverMeshDumper.h"


#include <Cleaver/Cleaver.h>
#include <Cleaver/InverseField.h>
#include <Cleaver/FloatField.h>


using namespace Cleaver;


CellFieldCleaverSimulator::CellFieldCleaverSimulator() : 
m_bounds(vec3::zero, vec3(1,1,1)),paddingDim(2,2,2),cellField(0)
{
	// no allocation
	minValue=1000000000.0;
	maxValue=-1000000000.0;        
}

CellFieldCleaverSimulator::~CellFieldCleaverSimulator()
{
	// no memory cleanup
}

BoundingBox CellFieldCleaverSimulator::bounds() const
{
	return m_bounds;
}

void CellFieldCleaverSimulator::setFieldDim(Dim3D _dim){
	fieldDim=_dim;    
	m_bounds.size=vec3(fieldDim.x,fieldDim.y,fieldDim.z);
}



float CellFieldCleaverSimulator::valueAt(float x, float y, float z) const
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
	}else if (includeCellTypesSet.find(cell->type)!=includeCellTypesSet.end()){
		return 2.0+cell->type;
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
cellFieldG(0),sim(0),potts(0),
xmlData(0),boundaryStrategy(0),automaton(0),cellInventoryPtr(0)
{
	meshOutputFormat="tetgen";
	outputMeshSurface=false;
	verbose=false;

}

CleaverMeshDumper::~CleaverMeshDumper() {
}


void CleaverMeshDumper::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	xmlData=_xmlData;

	potts = simulator->getPotts();
	cellInventoryPtr=& potts->getCellInventory();
	sim=simulator;
	cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	fieldDim=cellFieldG->getDim();


	simulator->registerSteerableObject(this);

	update(_xmlData,true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CleaverMeshDumper::extraInit(Simulator *simulator){
	//PUT YOUR CODE HERE
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CleaverMeshDumper::start(){

	//PUT YOUR CODE HERE

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CleaverMeshDumper::buildDolfinMesh(dolfin::Mesh & _mesh){

  
  dolfin::MeshEditor editor;
//   dolfin::CellType::tetrahedron;
  editor.open(_mesh, dolfin::CellType::triangle, 2, 2);
  editor.init_vertices(4);
  editor.init_cells(2);
  editor.add_vertex(0, 0.0, 0.0);
  editor.add_vertex(1, 1.0, 0.0);
  editor.add_vertex(2, 1.0, 1.0);
  editor.add_vertex(3, 0.0, 1.0);
  editor.add_cell(0, 0, 1, 2);
  editor.add_cell(1, 0, 2, 3);
  editor.close();
  
  cerr<<"Number of cells="<<_mesh.num_cells()<<endl;
  cerr<<"Number of vertices="<<_mesh.num_vertices()<<endl;
  
//   editor.open(mesh, 2, 2);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CleaverMeshDumper::buildDolfinMeshFromCleaver(dolfin::Mesh & _mesh,Cleaver::TetMesh & _cleaverMesh){
  dolfin::MeshEditor editor;
  
  cerr<<"_cleaverMesh.verts.size()="<<_cleaverMesh.verts.size()<<endl;
  cerr<<"_cleaverMesh.tets.size()="<<_cleaverMesh.tets.size()<<endl;
  
  editor.open(_mesh, dolfin::CellType::tetrahedron, 3, 3);
  editor.init_vertices(_cleaverMesh.verts.size());  
  editor.init_cells(_cleaverMesh.tets.size());
  
  //writing vertices to dolfin mesh 
  for(unsigned int i=0; i < _cleaverMesh.verts.size(); i++)
  {
    editor.add_vertex(i,_cleaverMesh.verts[i]->pos().x,_cleaverMesh.verts[i]->pos().y,_cleaverMesh.verts[i]->pos().z);      
  }
  
  //writing tetrahedrons to dolfin mesh 
  for(unsigned int i=0; i < _cleaverMesh.tets.size(); i++)
  {
    
    editor.add_cell(i,_cleaverMesh.verts[0]->tm_v_index,_cleaverMesh.verts[1]->tm_v_index,_cleaverMesh.verts[2]->tm_v_index,_cleaverMesh.verts[3]->tm_v_index);      
  }

  editor.close();
  
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CleaverMeshDumper::simulateCleaverMesh(){
	CellFieldCleaverSimulator cfcs;
	cfcs.setFieldDim(fieldDim);
	cfcs.setCellFieldPtr(cellFieldG);
	cfcs.setIncludeCellTypesSet(cellTypesSet);

	//bool verbose=true;
	//string outputFileName="cellfieldmesh";

	Cleaver::InverseField inverseField = Cleaver::InverseField(&cfcs);

	std::vector<Cleaver::ScalarField*> fields;

	fields.push_back(&cfcs);
	fields.push_back(&inverseField);

	Cleaver::Volume volume(fields);
	Cleaver::TetMesh *mesh = Cleaver::createMeshFromVolume(volume, verbose);

	//mesh->writeNodeEle(outputFileName, verbose);
	cerr<<"outputFileName="<<outputFileName<<endl;
	cerr<<"verbose="<<verbose<<endl;


    if(meshOutputFormat == "tetgen")
        mesh->writeNodeEle(outputFileName, verbose);
    else if(meshOutputFormat == "scirun")
        mesh->writePtsEle(outputFileName, verbose);
    else if(meshOutputFormat == "matlab")
        mesh->writeMatlab(outputFileName, verbose);


	//----------------------
	// Write Surface Files
	//----------------------
	if (outputMeshSurface){
		mesh->constructFaces();
		mesh->writePly(outputFileName, verbose);
	}

	

      dolfin::Mesh meshDolfin;	 
      buildDolfinMeshFromCleaver(meshDolfin,*mesh);
      
     delete mesh; 
//       buildDolfinMesh(meshDolfin);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CleaverMeshDumper::step(const unsigned int currentStep){
	if (! currentStep%10){
		simulateCleaverMesh();
	}
	//   //REPLACE SAMPLE CODE BELOW WITH YOUR OWN
	//CellInventory::cellInventoryIterator cInvItr;
	//CellG * cell;
	//   
	//   cerr<<"currentStep="<<currentStep<<endl;
	//for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )
	//{
	//	cell=cellInventoryPtr->getCell(cInvItr);
	//       cerr<<"cell.id="<<cell->id<<" vol="<<cell->volume<<endl;
	//   }

}


void CleaverMeshDumper::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	//PARSE XML IN THIS FUNCTION
	//For more information on XML parser function please see CC3D code or lookup XML utils API
	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
	cellTypesSet.clear();

	std::vector<std::string> typeNames;
	

	CC3DXMLElement * outputXMLElem=_xmlData->getFirstElement("OutputSpecification");
	if(outputXMLElem){
		outputFileName=outputXMLElem->getFirstElement("OutputFileNeme")->getText();
		if (outputXMLElem->getFirstElement("OutputMeshSurface")){
			outputMeshSurface=true;
		}

		if (outputXMLElem->getFirstElement("Verbose")){
			verbose=true;
			cerr<<"verbose="<<verbose<<endl;
		}


		if (outputXMLElem->getFirstElement("MeshOutputFormat")){
			meshOutputFormat=outputXMLElem->getFirstElement("MeshOutputFormat")->getText();
		}

		if (outputXMLElem->getFirstElement("IncludeCellTypes")){
			string celTypeStr=outputXMLElem->getFirstElement("IncludeCellTypes")->getText();
			parseStringIntoList( celTypeStr, typeNames , ",");
			for (int i = 0 ; i < typeNames.size(); ++i){
				cellTypesSet.insert(automaton->getTypeId(typeNames[i]));
				
			}


			for (set<unsigned char>::iterator sitr = cellTypesSet.begin() ; sitr!=cellTypesSet.end() ;++sitr){
				cerr<<"INCLUDIG CELL TYPE="<<(int)*sitr<<endl;
			}


		}
	}


	//boundaryStrategy has information aobut pixel neighbors 
	boundaryStrategy=BoundaryStrategy::getInstance();

}

std::string CleaverMeshDumper::toString(){
	return "CleaverMeshDumper";
}

std::string CleaverMeshDumper::steerableName(){
	return toString();
}

