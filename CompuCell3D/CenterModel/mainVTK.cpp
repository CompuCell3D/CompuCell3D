//#include <stdio.h>

#include <fstream>
#include <string>
#include <iostream>
#include <cassert>

#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkConeSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCellArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>
#include <vtkCamera.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
#include <vtkSphereSource.h>
#include <vtkGlyph3D.h>

#include <Windows.h>
#include <Components/CellFactoryCM.h>
#include <Components/SimulationBox.h>
#include <Components/CellInventoryCM.h>

using std::string;
using namespace  CenterModel;

struct CLArguments{
	std::string fileName;
};

CLArguments parceCommandLine(int argc, char *argv[]){
	if(argc<=2){
		fprintf(stderr, "Not enough command line arguments\n");
		exit(EXIT_FAILURE);
	}

	CLArguments res;

	for(int i=1; i<argc; ++i){
		if(strcmp(argv[i], "-f")==0){
			res.fileName=argv[++i];
		}
	}

	return res;

}


//Parsing a string to make a cell
//I'd bettef use std::unique_ptr<CellCM> here,
//but for the sake of backward compatability with VS2008 I used plain pointers
CellCM * parseString(std::istream &is, CellFactoryCM &cf){
	assert(is);
	float x,y,z,r;
	is>>x>>y>>z>>r;
	
	//cout<<x<<" "<<y<<" "<<z<<" "<<r<<"\n";

	if(is.fail())
		return NULL;
	
	CellCM *cellTmp=cf.createCellCM(x,y,z);
    cellTmp->interactionRadius=r;
   
	return cellTmp;
}

std::pair<Vector3, Vector3> detectBoundingBox(CellInventoryCM const &ci){
	const Vector3::precision_t maxReal=std::numeric_limits<Vector3::precision_t>::max();
	Vector3 minDim(maxReal, maxReal, maxReal), maxDim(-maxReal,-maxReal,-maxReal);

	for (CellInventoryCM::cellInventoryIteratorConst cit=ci.cellInventoryBegin() ; cit!=ci.cellInventoryEnd(); ++cit){
		CellCM const* cell=cit->second;
		Vector3 const &cellPos=cell->position;
		minDim.SetMax(cellPos);
		maxDim.SetMin(cellPos);
	}

	return make_pair(minDim, maxDim);

}


CellInventoryCM *makeCellInventory(std::istream &is, CellFactoryCM &cf)
{

	string str;

	
    CellInventoryCM *ci=new CellInventoryCM();
	
	ci->setCellFactory(&cf);

	while(!is.eof()){
		CellCM *cm=parseString(is, cf);

		if(cm==NULL)
			break;
		ci->addToInventory(cm);
	}
	

	return ci;
}

void updateCellsLookup(CellInventoryCM  &ci, SimulationBox &sb){
	for (CellInventoryCM::cellInventoryIterator cit=ci.cellInventoryBegin() ; cit!=ci.cellInventoryEnd(); ++cit){
		CellCM * cell=cit->second;
		sb.updateCellLookup(cell);
	}
}


//generate vtkPolyData based on CellInventory
//TODO: needs binding instead
vtkSmartPointer<vtkPolyData> genPolydata(CellInventoryCM const &ci){
	vtkSmartPointer<vtkPolyData> res=vtkSmartPointer<vtkPolyData>::New();

	//vtkSmartPointer<vtkCellArray> vertices =  vtkSmartPointer<vtkCellArray>::New();
 	vtkSmartPointer<vtkPoints> points =vtkSmartPointer<vtkPoints>::New();

	// Setup the colors array
    //vtkSmartPointer<vtkUnsignedCharArray> colors =vtkSmartPointer<vtkUnsignedCharArray>::New();
	//colors->SetNumberOfComponents(3);
	//colors->SetName("Colors");

	vtkSmartPointer<vtkDoubleArray> radii =vtkSmartPointer<vtkDoubleArray>::New();
//	radii->SetNumberOfComponents(1);
	radii->SetName("Radii");


   for (CellInventoryCM::cellInventoryIteratorConst cit=ci.cellInventoryBegin() ; cit!=ci.cellInventoryEnd(); ++cit){
		CellCM const* cell=cit->second;
		Vector3 const &cellPos=cell->position;
		vtkIdType type=points->InsertNextPoint(cellPos.fX, cellPos.fY, cellPos.fZ);
//		vertices->InsertNextCell(1, &type);

		//unsigned char color[3] = {rand()/(double)RAND_MAX*255, rand()/(double)RAND_MAX*255, rand()/(double)RAND_MAX*255};
		//double scalar=rand()/(double)RAND_MAX*1265.;
		double scalar=cell->interactionRadius;
		radii->InsertNextTupleValue(&scalar);

		//aCone->GetProperty()->SetColor(rand()/(double)RAND_MAX,rand()/(double)RAND_MAX,rand()/(double)RAND_MAX); // cone color blue

	}

	res->SetPoints(points);
	//res->SetVerts(vertices);
	res->GetPointData()->SetScalars(radii);

	return res;
}

double detectMaxRadius(vtkPolyData *const pointsPolydata)
{
	double maxRad=0;
	vtkDataArray *arr=pointsPolydata->GetPointData()->GetScalars();
	for(int i=0; i<pointsPolydata->GetNumberOfPoints(); ++i){
		double rad=*arr->GetTuple(i);
		maxRad=std::max(rad, maxRad);
	}
	return maxRad;
}


void goVTK(CellInventoryCM const &ci){
	vtkSmartPointer<vtkPolyData> pointsPolydata =genPolydata(ci);
	cout<<pointsPolydata->GetNumberOfPoints()<<" points are put into a vtkPolydata object"<<endl;

	double maxRad=detectMaxRadius(pointsPolydata);
	cout<<"max cell's radius: "<<maxRad<<endl;


	vtkSmartPointer<vtkSphereSource> sphereSource=vtkSphereSource::New();
	sphereSource->SetThetaResolution(20);
	sphereSource->SetPhiResolution(20);
	
	
	vtkSmartPointer<vtkGlyph3D> glyph=vtkGlyph3D::New();
	glyph->SetScaleModeToScaleByScalar();
	glyph->SetSourceConnection(sphereSource->GetOutputPort());
	glyph->SetInput(pointsPolydata);
	glyph->Update();


	// map to graphics library 
    vtkSmartPointer<vtkPolyDataMapper> map = vtkPolyDataMapper::New(); 
	map->SetInputConnection(glyph->GetOutputPort());

	// actor coordinates geometry, properties, transformation 
	vtkSmartPointer<vtkActor> actor = vtkActor::New(); 
	actor->SetMapper(map); 
	//actor->GetProperty()->SetPointSize(10);
	actor->GetProperty()->SetColor(0,0,1); 

	vtkSmartPointer<vtkRenderWindow> renderWindow =vtkSmartPointer<vtkRenderWindow>::New();
	vtkSmartPointer<vtkRenderer> renderer = vtkRenderer::New(); 
	renderWindow->AddRenderer(renderer);

	renderer->AddActor(actor); 
	renderer->SetBackground(1,1,1); // Background color white


	vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkRenderWindowInteractor::New(); 
	iren->SetRenderWindow(renderWindow);

	vtkInteractorStyleTrackballCamera *style =  vtkInteractorStyleTrackballCamera::New();
	iren->SetInteractorStyle(style);
	
	//actor->SetOrigin(center);

	renderWindow->SetSize(640, 480);

	renderWindow->Render();
	iren->Start();
}

int main(int argc, char *argv[]){
	CLArguments clArgs=parceCommandLine(argc, argv);
	cout<<"Reading cells from file: "<<clArgs.fileName<<endl;
	
	char buffer[250];
	GetCurrentDirectory(250, buffer);

	cout<<buffer<<endl;

	fstream ifile(clArgs.fileName);
	if(!ifile.is_open()){
		cerr<<"Can not open file "<<clArgs.fileName<<endl;
		return EXIT_FAILURE;
	}

	CellFactoryCM *cf= new CellFactoryCM();
	
	//read cells from file and put them into the inventory 
	CellInventoryCM  *ci=makeCellInventory(ifile, *cf);//allocated with new withing a function
	cout<<ci->getSize()<<" cells have been added to the inventory"<<endl;

	
	std::pair<Vector3, Vector3> bBox=detectBoundingBox(*ci);
	cout<<"cells are enclosed in a bounding box "<<bBox.first<<"x"<<bBox.second<<endl;

	//assume that we start from zero and enlarge max dimension a bit...
	Vector3 simBoxDim(bBox.second[0]*1.1, bBox.second[1]*1.1, bBox.second[2]*1.1);
	cout<<"setting simulation box to"<<simBoxDim<<endl;

	SimulationBox *sb=new SimulationBox();
	sb->setBoxSpatialProperties(simBoxDim,Vector3(1.01,1.01,1.01));//HACK: spacing is hardcoded

	updateCellsLookup(*ci, *sb);

	cf->setSimulationBox(sb);

	cout<<"CompuCell Center Model is ready, setting up VTK pipeline..."<<endl;


	goVTK(*ci);
	

	delete sb;
	delete ci;
	delete cf;

}