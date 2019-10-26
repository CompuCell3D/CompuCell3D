

#include <CompuCell3D/CC3D.h>        

using namespace CompuCell3D;



#include "LocalAntiFragmentationPlugin.h"





LocalAntiFragmentationPlugin::LocalAntiFragmentationPlugin():

pUtils(0),

lockPtr(0),

xmlData(0) ,

cellFieldG(0),

boundaryStrategy(0)

{}



LocalAntiFragmentationPlugin::~LocalAntiFragmentationPlugin() {

    pUtils->destroyLock(lockPtr);

    delete lockPtr;

    lockPtr=0;

}



void LocalAntiFragmentationPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    xmlData=_xmlData;

    sim=simulator;

    potts=simulator->getPotts();

    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

    

    pUtils=sim->getParallelUtils();

    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;

    pUtils->initLock(lockPtr); 

   

   update(xmlData,true);

   

    

    //potts->registerEnergyFunctionWithName(this,"LocalAntiFragmentation");

   potts->registerConnectivityConstraint(this); // we give it a special status to run it only when really needed 

    potts->registerCellGChangeWatcher(this);    

    

    

    simulator->registerSteerableObject(this);

}



void LocalAntiFragmentationPlugin::extraInit(Simulator *simulator){

    

}



            

void LocalAntiFragmentationPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) 

{

    
    //This function will be called after each succesful pixel copy - field3DChange does usuall ohusekeeping tasks to make sure state of cells, and state of the lattice is uptdate
    if (newCell){
        //PUT YOUR CODE HERE
    }else{
        //PUT YOUR CODE HERE
    }

    if (oldCell){
        //PUT YOUR CODE HERE
    }else{
        //PUT YOUR CODE HERE
    }

		

}







double LocalAntiFragmentationPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {	

	//return (this->*changeEnergyFcnPtr)(pt, newCell, oldCell);

	

    double energy = 0;
    if (oldCell){
        //PUT YOUR CODE HERE
    }else{
        //PUT YOUR CODE HERE
    }
    
    if(newCell){
        //PUT YOUR CODE HERE
    }else{
        //PUT YOUR CODE HERE
    }
    
    return energy;    
}            





void LocalAntiFragmentationPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

    //PARSE XML IN THIS FUNCTION

    //For more information on XML parser function please see CC3D code or lookup XML utils API

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

   set<unsigned char> cellTypesSet;



    CC3DXMLElement * exampleXMLElem=_xmlData->getFirstElement("Example");

    if (exampleXMLElem){

        double param=exampleXMLElem->getDouble();

        cerr<<"param="<<param<<endl;

        if(exampleXMLElem->findAttribute("Type")){

            std::string attrib=exampleXMLElem->getAttribute("Type");

            // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double

            cerr<<"attrib="<<attrib<<endl;

        }

    }

    

    //boundaryStrategy has information aobut pixel neighbors 

    boundaryStrategy=BoundaryStrategy::getInstance();



}





std::string LocalAntiFragmentationPlugin::toString(){

    return "LocalAntiFragmentation";

}





std::string LocalAntiFragmentationPlugin::steerableName(){

    return toString();

}



double LocalAntiFragmentationPlugin::localConectivity(Point3D * changePixel, Point3D * flipNeighbor)
{
	
	return (this->*localConnectFcnPtr)( changePixel,  flipNeighbor);
	
	//return 0.0;//place holder
}

double LocalAntiFragmentationPlugin::localConectivity_2D(Point3D *changePixel, Point3D *flipNeighbor)
{



	CellG *changeCell = cellFieldG->get(*changePixel);

	if (!changeCell)return true;//if it is medium we don't care


	CellG *flipCell = cellFieldG->get(*flipNeighbor);

	CellG *nCell = 0;

	Neighbor neighbor;

	BoundaryStrategy * boundaryStrategy = BoundaryStrategy::getInstance();



	NumericalUtils * numericalUtils;

	Dim3D fieldDim = getCellFieldG()->getDim();

	// maximum pixel neighbor index for 1st and second neighbors
	unsigned int fNeumanNeighIdx = BoundaryStrategy::getInstance()->getMaxNeighborIndexFromNeighborOrder(1);
	unsigned int fMooreNeighIdx = BoundaryStrategy::getInstance()->getMaxNeighborIndexFromNeighborOrder(2);

	//compute neigbor pixels ownership

	int same_owner = 0;


	for (unsigned int nIdx = 0; nIdx <= fNeumanNeighIdx; nIdx++)
	{
		neighbor = boundaryStrategy->getNeighborDirect(*changePixel, nIdx);

		if (!neighbor.distance) continue; //if distance == 0 returned neighbor is invalid

		else if (changeCell == cellFieldG->get(neighbor.pt)) ++same_owner;

	}

	//compute local conectedness
	double localConnected = 1.0; // 0.0 means false, all else true

	if (same_owner >= 4) return 1.0; //(true). if all neigbors are the same as the one being changed it's going to be always connected (shouldn't be >4 ever but, you know, bugs)


	if (same_owner > 1)
	{

		//Dim3D fieldDim = getCellFieldG()->getDim(); //need to put this in a better place

		bool N, S, E, W;
		bool NE, NW, SE, SW;


		//N, S, E, W:

		Point3D ptN = *changePixel;

		ptN.y += 1;


		ptN = *changePixel + CompuCell3D::distanceVectorInvariant(ptN, *changePixel, fieldDim);

		N = (changeCell == cellFieldG->get(ptN));

		//______________

		Point3D ptS = *changePixel;

		ptS.y -= 1;


		//Dim3D fieldDim = getCellFieldG()->getDim();

		/*Point3D dN = CompuCell3D::distanceVectorInvariant(ptN, * changePixel, fieldDim);

		ptN = *changePixel + dN;*/

		ptS = *changePixel + CompuCell3D::distanceVectorInvariant(ptS, *changePixel, fieldDim);

		S = (changeCell == cellFieldG->get(ptS));

		//______________

		Point3D ptE = *changePixel;

		ptE.x += 1;

		ptE = *changePixel + CompuCell3D::distanceVectorInvariant(ptE, *changePixel, fieldDim);

		E = (changeCell == cellFieldG->get(ptE));

		//______________

		Point3D ptW = *changePixel;

		ptW.x -= 1;

		ptW = *changePixel + CompuCell3D::distanceVectorInvariant(ptW, *changePixel, fieldDim);

		W = (changeCell == cellFieldG->get(ptW));
		
		
		//______________
		//NE, NW, SE, SW:

		Point3D ptNE = *changePixel;

		ptNE.y += 1;
		ptNE.x += 1;

		ptNE = *changePixel + CompuCell3D::distanceVectorInvariant(ptNE, *changePixel, fieldDim);

		NE = (changeCell == cellFieldG->get(ptNE));

		//______________

		Point3D ptNW = *changePixel;

		ptNW.y += 1;
		ptNW.x -= 1;

		ptNW = *changePixel + CompuCell3D::distanceVectorInvariant(ptNW, *changePixel, fieldDim);

		NW = (changeCell == cellFieldG->get(ptNW));

		//______________

		Point3D ptSE = *changePixel;

		ptSE.y -= 1;
		ptSE.x += 1;

		ptSE = *changePixel + CompuCell3D::distanceVectorInvariant(ptSE, *changePixel, fieldDim);

		SE = (changeCell == cellFieldG->get(ptSE));

		//______________

		Point3D ptSW = *changePixel;

		ptSW.y -= 1;
		ptSW.x -= 1;

		ptSW = *changePixel + CompuCell3D::distanceVectorInvariant(ptSW, *changePixel, fieldDim);

		SE = (changeCell == cellFieldG->get(ptSW));


		if (
			(same_owner == 2 
					&& !((N&&E&&NE) || (N&&W&&NW) || (S&&E&&SE) || (S&&W&&SW)))

		||  (same_owner == 3 
					&& !((S || (NE&&NW)) && (E || (NW&&SW)) && (N || (SE&&SW)) && (W || (SE&&NE)))))
		{
			localConnected = 0.0;
		}
	}


	return localConnected;



}