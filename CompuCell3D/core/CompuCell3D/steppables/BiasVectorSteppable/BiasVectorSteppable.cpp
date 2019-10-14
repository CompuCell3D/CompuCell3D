



#include <CompuCell3D/CC3D.h>



using namespace CompuCell3D;



using namespace std;



#include "BiasVectorSteppable.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>


BiasVectorSteppable::BiasVectorSteppable() : cellFieldG(0),sim(0),potts(0),xmlData(0),boundaryStrategy(0),automaton(0),cellInventoryPtr(0){}



BiasVectorSteppable::~BiasVectorSteppable() {

}





void BiasVectorSteppable::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	cerr << "got into biasvec step" << endl;

  xmlData=_xmlData;

  

  potts = simulator->getPotts();

  cellInventoryPtr=& potts->getCellInventory();

  sim=simulator;

  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

  fieldDim=cellFieldG->getDim();



  
  //potts->getCellFactoryGroupPtr()->registerClass(&biasVectorSteppableDataAccessor);
  simulator->registerSteerableObject(this);



  update(_xmlData,true);



}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void BiasVectorSteppable::extraInit(Simulator *simulator){

    //PUT YOUR CODE HERE

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BiasVectorSteppable::start(){



  //PUT YOUR CODE HERE



}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
STEPPERS
*/

void BiasVectorSteppable::step(const unsigned int currentStep){

	return (this->*stepFcnPtr)(currentStep);

}



//rename this fncs to white noise 2d/3d
void CompuCell3D::BiasVectorSteppable::step_3d(const unsigned int currentStep)
{
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell = 0;
	
	//cout << "in bias vector 3d step" << endl;

	//cerr << "currentStep=" << currentStep << endl;

	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)
	{


		cell = cellInventoryPtr->getCell(cInvItr);

		vector<double> noise = BiasVectorSteppable::noise_vec_generator();

		cell->biasVecX = noise[0];
		cell->biasVecY = noise[1];
		cell->biasVecZ = noise[2];

	}
}

//pure white random change bias:
void CompuCell3D::BiasVectorSteppable::step_2d_x(const unsigned int currentStep)
{
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell = 0;

	
	//cout << "in bias vector step" << endl;

	//cerr << "currentStep=" << currentStep << endl;

	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)
	{

		cell = cellInventoryPtr->getCell(cInvItr);

		vector<double> noise = BiasVectorSteppable::noise_vec_generator();


		cell->biasVecX = 0;
		cell->biasVecY = noise[0];
		cell->biasVecZ = noise[1];
	}

}

//pure white random change bias:
void CompuCell3D::BiasVectorSteppable::step_2d_y(const unsigned int currentStep)
{
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell = 0;


	//cout << "in bias vector step" << endl;

	//cerr << "currentStep=" << currentStep << endl;

	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)
	{

		cell = cellInventoryPtr->getCell(cInvItr);

		vector<double> noise = BiasVectorSteppable::noise_vec_generator();


		cell->biasVecX = noise[0];
		cell->biasVecY = 0;
		cell->biasVecZ = noise[1];
	}
}
//pure white random change bias:
void CompuCell3D::BiasVectorSteppable::step_2d_z(const unsigned int currentStep)
{
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell = 0;


	//cout << "in bias vector step" << endl;

	//cerr << "currentStep=" << currentStep << endl;

	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)
	{


		cell = cellInventoryPtr->getCell(cInvItr);

		vector<double> noise = BiasVectorSteppable::noise_vec_generator();

		cell->biasVecX = noise[0];
		cell->biasVecY = noise[1];
		cell->biasVecZ = 0;
	}
}


void BiasVectorSteppable::step_momentum_bias(const unsigned int currentStep)
{
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell = 0;

	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)
	{
		cell = cellInventoryPtr->getCell(cInvItr);
		double alpha = biasMomenParamVec[cell->type].momentumAlpha;

		gen_momentum_bias(alpha, cell);

	}
}



//==========================================================
/*
MOMENTUM BIAS GENERATORS
*/

void BiasVectorSteppable::gen_momentum_bias(const double alpha, CellG * cell) {

	return (this->*momGenFcnPtr)(alpha,cell);

}





void BiasVectorSteppable::gen_momentum_bias_3d(const double alpha, CellG *cell)
{
	vector<double> noise = BiasVectorSteppable::noise_vec_generator();

	cell->biasVecX = alpha*cell->biasVecX + (1 - alpha)*noise[0];
	cell->biasVecY = alpha*cell->biasVecY + (1 - alpha)*noise[1];
	cell->biasVecZ = alpha*cell->biasVecZ + (1 - alpha)*noise[2];

}


void BiasVectorSteppable::gen_momentum_bias_2d_x(const double alpha, CellG *cell)
{
	vector<double> noise = BiasVectorSteppable::noise_vec_generator();

	cell->biasVecX = 0;
	cell->biasVecY = alpha*cell->biasVecY + (1 - alpha)*noise[0];
	cell->biasVecZ = alpha*cell->biasVecZ + (1 - alpha)*noise[1];
}

void BiasVectorSteppable::gen_momentum_bias_2d_y(const double alpha, CellG *cell)
{
	vector<double> noise = BiasVectorSteppable::noise_vec_generator();

	cell->biasVecX = alpha*cell->biasVecX + (1 - alpha)*noise[0];
	cell->biasVecY = 0;
	cell->biasVecZ = alpha*cell->biasVecZ + (1 - alpha)*noise[1];
}

void BiasVectorSteppable::gen_momentum_bias_2d_z(const double alpha, CellG *cell)
{
	vector<double> noise = BiasVectorSteppable::noise_vec_generator();

	cell->biasVecX = alpha*cell->biasVecX + (1 - alpha)*noise[0];
	cell->biasVecY = alpha*cell->biasVecY + (1 - alpha)*noise[1];
	cell->biasVecZ = 0;
}



/*
NOISE VECTORS GENERATORS
*/

vector<double> BiasVectorSteppable::noise_vec_generator()
{
	return (this->*noiseFcnPtr)();
}


//creates unitary 2d white noise vector
vector<double> BiasVectorSteppable::white_noise_2d()
{
	// cout << "in the 2d white noise method" << endl;

	BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();

	double angle = rand->getRatio() * 2 * M_PI;
	double x0 = std::cos(angle);
	double x1 = std::sin(angle);

	vector<double> noise{ x0,x1 };
	
	return noise;
}


vector<double> BiasVectorSteppable::white_noise_3d()
{
	//cout << "in the 3d white noise method" << endl;
	BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();

	//method for getting random unitary vector in sphere from Marsaglia 1972
	//example and reason for not using a uniform distribution
	//can be found @ mathworld.wolfram.com/SpherePointPicking.html

	double tx = 2 * rand->getRatio() - 1;
	double ty = 2 * rand->getRatio() - 1;

	double dist_sqrd = (tx*tx + ty*ty);
	/*cerr << "in the 3d step method" << endl;*/

	while (dist_sqrd >= 1)
	{
		tx = 2 * rand->getRatio() - 1;
		ty = 2 * rand->getRatio() - 1;


		dist_sqrd = tx*tx + ty*ty;

	}

	if (dist_sqrd < 1)
	{
		double x = 2 * tx * std::sqrt(1 - tx*tx - ty*ty);
		double y = 2 * ty * std::sqrt(1 - tx*tx - ty*ty);
		double z = 1 - 2 * (tx*tx + ty*ty);
		vector<double> noise{ x, y, z };

		return noise;
	}
	//TODO: some sort of catching for infinite loops
}




void BiasVectorSteppable::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){



    //PARSE XML IN THIS FUNCTION

    //For more information on XML parser function please see CC3D code or lookup XML utils API

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

   set<unsigned char> cellTypesSet;


	/*
	Example xml:

	A)
	______________________
	<BiasChange Type ="momentum">
	<BiasChangeParameters Alpha="10" CellType="cell1"/>
	...
	</BiasChange>

	B)
	____________________
	<BiasChange>
	</BiasChange>

	*/

    
	if (!_xmlData)
	{
		biasType = WHITE;
	}
	else if (_xmlData->findElement("BiasChange"))
	{
		string biasChangeStr = _xmlData->getAttribute("BiasChange");// xml style will be <BiasChange="type of change">, 
																	//might have trouble with the muExpressions?
		transform(biasChangeStr.begin(), biasChangeStr.end(), biasChangeStr.begin(), ::tolower);
		if (biasChangeStr == "white")// b = white noise
		{
			biasType = WHITE;
		}
		else if (biasChangeStr == "momentum")// b(t+1) = a*b(t) + (1-a)*noise
		{
			biasType = MOMENTUM;
		}
		else if (biasChangeStr == "manual")// for changing b in python
		{
			biasType = MANUAL;
		}
		else if (biasChangeStr == "custom")// for muExpressions
		{
			biasType = CUSTOM;
		}
		else
		{
			throw std::invalid_argument("Invalid bias change type in BiasChange");
		}
	}
	else
	{
		biasType = WHITE;
	}








    //boundaryStrategy has information aobut pixel neighbors 

    boundaryStrategy=BoundaryStrategy::getInstance();

	if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z == 1)
	{
		noiseType = VEC_GEN_WHITE2D;
	}
	else
	{
		noiseType = VEC_GEN_WHITE3D;
	}


	switch (noiseType)
	{
	case VEC_GEN_WHITE2D:
		noiseFcnPtr = &BiasVectorSteppable::white_noise_2d;
		break;
	case VEC_GEN_WHITE3D:
		noiseFcnPtr = &BiasVectorSteppable::white_noise_3d;
		break;
	default:
		noiseFcnPtr = &BiasVectorSteppable::white_noise_3d;
		break;
	}




	if (fieldDim.x == 1)
	{
		fieldType = FTYPE2DX;
	}
	else if (fieldDim.y == 1)
	{
		fieldType = FTYPE2DY;
	}
	else if (fieldDim.z == 1)
	{
		fieldType = FTYPE2DZ;
	}
	else
	{
		fieldType = FTYPE3D;
	}





	vector<int> typeIdVec;

	vector<BiasMomenParam> momentumParamVecTmp;

	CC3DXMLElementList paramVec;

	int maxTypeID;

	vector<int>::iterator pos;

	switch (biasType)
	{
	case WHITE:
		switch (fieldType)
		{
		case FTYPE3D:
			stepFcnPtr = &BiasVectorSteppable::step_3d;
			break;
		case FTYPE2DX:
			stepFcnPtr = &BiasVectorSteppable::step_2d_x;
			break;
		case FTYPE2DY:
			stepFcnPtr = &BiasVectorSteppable::step_2d_y;
			break;
		case FTYPE2DZ:
			stepFcnPtr = &BiasVectorSteppable::step_2d_z;
			break;
		default:
			stepFcnPtr = &BiasVectorSteppable::step_3d;
			break;
		}
		break;
	case MOMENTUM:
		//For now I'll only do by cell type.
		
		stepFcnPtr = &BiasVectorSteppable::step_momentum_bias;

		biasMomenParamVec.clear();

		/*vector<int> typeIdVec;*/

		/*vector<BiasMomenParam> momentumParamVecTmp;*/

		/*CC3DXMLElementList*/ paramVec = _xmlData->getElements("BiasChangeParameters");

		for (int i = 0; i < paramVec.size(); ++i)
		{
			BiasMomenParam biasParam;

			biasParam.momentumAlpha = paramVec[i]->getAttributeAsDouble("Alpha");
			biasParam.typeName = paramVec[i]->getAttribute("CellType");

			cerr << "automaton=" << automaton << endl;

			typeIdVec.push_back(automaton->getTypeId(biasParam.typeName));

			//momentumParamVecTmp.push_back(biasParam.typeName);
			momentumParamVecTmp.push_back(biasParam);
		}

		/*vector<int>::iterator*/ pos = max_element(typeIdVec.begin(), typeIdVec.end());
		/*int*/ maxTypeID = *pos;

		biasMomenParamVec.assign(maxTypeID + 1, BiasMomenParam());

		for (int i = 0; i < momentumParamVecTmp.size(); ++i)
		{
			biasMomenParamVec[typeIdVec[i]] = momentumParamVecTmp[i];
		}

		switch (fieldType)
		{
		case CompuCell3D::BiasVectorSteppable::FTYPE3D:
			momGenFcnPtr = &BiasVectorSteppable::gen_momentum_bias_3d;
			break;
		case CompuCell3D::BiasVectorSteppable::FTYPE2DX:
			momGenFcnPtr = &BiasVectorSteppable::gen_momentum_bias_2d_x;
			break;
		case CompuCell3D::BiasVectorSteppable::FTYPE2DY:
			momGenFcnPtr = &BiasVectorSteppable::gen_momentum_bias_2d_y;
			break;
		case CompuCell3D::BiasVectorSteppable::FTYPE2DZ:
			momGenFcnPtr = &BiasVectorSteppable::gen_momentum_bias_2d_z;
			break;
		default:
			momGenFcnPtr = &BiasVectorSteppable::gen_momentum_bias_3d;
			break;
		}
		break;
	
	
	
	default:
		switch (fieldType)
		{
		case FTYPE3D:
			stepFcnPtr = &BiasVectorSteppable::step_3d;
			break;
		case FTYPE2DX:
			stepFcnPtr = &BiasVectorSteppable::step_2d_x;
			break;
		case FTYPE2DY:
			stepFcnPtr = &BiasVectorSteppable::step_2d_y;
			break;
		case FTYPE2DZ:
			stepFcnPtr = &BiasVectorSteppable::step_2d_z;
			break;
		default:
			stepFcnPtr = &BiasVectorSteppable::step_3d;
			break;
		}
		break;

	}


/*
	switch (fieldType)
	{
	case FTYPE3D:
		stepFcnPtr = &BiasVectorSteppable::step_3d;
		break;
	case FTYPE2DX:
		stepFcnPtr = &BiasVectorSteppable::step_2d_x;
		break;
	case FTYPE2DY:
		stepFcnPtr = &BiasVectorSteppable::step_2d_y;
		break;
	case FTYPE2DZ:
		stepFcnPtr = &BiasVectorSteppable::step_2d_z;
		break;
	default:
		stepFcnPtr = &BiasVectorSteppable::step_3d;
	}*/

}



std::string BiasVectorSteppable::toString(){

   return "BiasVectorSteppable";

}



std::string BiasVectorSteppable::steerableName(){

   return toString();

}

        

