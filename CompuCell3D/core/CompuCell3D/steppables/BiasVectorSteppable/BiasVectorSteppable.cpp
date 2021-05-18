



#include <CompuCell3D/CC3D.h>



using namespace CompuCell3D;



using namespace std;



#include "BiasVectorSteppable.h"

#define _USE_MATH_DEFINES
#include <math.h>


BiasVectorSteppable::BiasVectorSteppable() : cellFieldG(0),sim(0),potts(0),xmlData(0),boundaryStrategy(0),automaton(0),cellInventoryPtr(0){}



BiasVectorSteppable::~BiasVectorSteppable() {
	pUtils->destroyLock(lockPtr);

	delete lockPtr;

	lockPtr = 0;
}





void BiasVectorSteppable::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

//cerr << "got into biasvec step" << endl;

  xmlData=_xmlData;

  

  potts = simulator->getPotts();

  cellInventoryPtr=& potts->getCellInventory();

  sim=simulator;

  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

  fieldDim=cellFieldG->getDim();



  
  //potts->getCellFactoryGroupPtr()->registerClass(&biasVectorSteppableDataAccessor);
  simulator->registerSteerableObject(this);

  pUtils = sim->getParallelUtils();

  lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;

  pUtils->initLock(lockPtr);



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


//vector<double> BiasVectorSteppable::

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
STEPPERS
*/

void BiasVectorSteppable::step(const unsigned int currentStep){

	return (this->*stepFcnPtr)(currentStep);

}



//rename this fncs to white noise 2d/3d
void CompuCell3D::BiasVectorSteppable::step_white_3d(const unsigned int currentStep)
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
void CompuCell3D::BiasVectorSteppable::step_white_2d_x(const unsigned int currentStep)
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
void CompuCell3D::BiasVectorSteppable::step_white_2d_y(const unsigned int currentStep)
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
void CompuCell3D::BiasVectorSteppable::step_white_2d_z(const unsigned int currentStep)
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

	/*std::ofstream func_name;
	func_name.open("function.txt");
	func_name << "in gen 2dz white" << std::endl;
	func_name.close();*/
}


void BiasVectorSteppable::step_persistent_bias(const unsigned int currentStep)
{
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell = 0;

	/*std::ofstream alpha_test;
	alpha_test.open("alpha_test.txt", std::ios_base::app);*/

	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)
	{
		cell = cellInventoryPtr->getCell(cInvItr);
		double alpha = biasPersistParamVec[cell->type].persistentAlpha;
		//alpha_test << alpha<<std::endl;
		gen_persistent_bias(alpha, cell);

	}
	//alpha_test.close();
}



//==========================================================
/*
PERSISTENT BIAS GENERATORS
*/

void BiasVectorSteppable::gen_persistent_bias(const double alpha, CellG * cell) {

	return (this->*perGenFcnPtr)(alpha,cell);

}


void BiasVectorSteppable::output_test(const double alpha, const CellG *cell, const vector<double> noise)
{
	std::ofstream vec_alpha_test;
	vec_alpha_test.open("vec_alpha_test.txt");// , std::ios_base::app);
	vec_alpha_test << noise[0] << ',' << noise[1] << ',' << noise[2] << std::endl
		<< (1 - alpha)*noise[0] << ',' << (1 - alpha)*noise[1] << ',' << (1 - alpha)*noise[2] << std::endl
		<< alpha*cell->biasVecX << ',' << alpha*cell->biasVecY << ',' << alpha*cell->biasVecZ << std::endl
		<< alpha*cell->biasVecX + (1 - alpha)*noise[0] << ',' << alpha*cell->biasVecY + (1 - alpha)*noise[1] << ',' << alpha*cell->biasVecZ + (1 - alpha)*noise[2] << std::endl
		<< "========================================="<<std::endl;
	return;
}


void BiasVectorSteppable::gen_persistent_bias_3d(const double alpha, CellG *cell)
{
	vector<double> noise = BiasVectorSteppable::noise_vec_generator();

	cell->biasVecX = alpha*cell->biasVecX + (1 - alpha)*noise[0];
	cell->biasVecY = alpha*cell->biasVecY + (1 - alpha)*noise[1];
	cell->biasVecZ = alpha*cell->biasVecZ + (1 - alpha)*noise[2];
	//output_test(alpha, cell, noise);

	/*std::ofstream func_name;
	func_name.open("function.txt");
	func_name << "in gen 3d" << std::endl;
	func_name.close();*/

}


void BiasVectorSteppable::gen_persistent_bias_2d_x(const double alpha, CellG *cell)
{
	vector<double> noise = BiasVectorSteppable::noise_vec_generator();

	cell->biasVecX = 0;
	cell->biasVecY = alpha*cell->biasVecY + (1 - alpha)*noise[0];
	cell->biasVecZ = alpha*cell->biasVecZ + (1 - alpha)*noise[1];
	//output_test(alpha, cell, noise);

	/*std::ofstream func_name;
	func_name.open("function.txt");
	func_name << "in gen 2dx" << std::endl;
	func_name.close();*/
}

void BiasVectorSteppable::gen_persistent_bias_2d_y(const double alpha, CellG *cell)
{
	vector<double> noise = BiasVectorSteppable::noise_vec_generator();

	cell->biasVecX = alpha*cell->biasVecX + (1 - alpha)*noise[0];
	cell->biasVecY = 0;
	cell->biasVecZ = alpha*cell->biasVecZ + (1 - alpha)*noise[1];
	//output_test(alpha, cell, noise);

	/*std::ofstream func_name;
	func_name.open("function.txt");
	func_name << "in gen 2dy" << std::endl;
	func_name.close();*/
}

void BiasVectorSteppable::gen_persistent_bias_2d_z(const double alpha, CellG *cell)
{
	vector<double> noise = BiasVectorSteppable::noise_vec_generator();

	cell->biasVecX = alpha*cell->biasVecX + (1 - alpha)*noise[0];
	cell->biasVecY = alpha*cell->biasVecY + (1 - alpha)*noise[1];
	cell->biasVecZ = 0;
	//output_test(alpha, cell, noise);

	/*std::ofstream func_name;
	func_name.open("function.txt");
	func_name << "in gen 2d_z" << std::endl;
	func_name.close();*/
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

	RandomNumberGenerator *rand = sim->getRandomNumberGeneratorInstance();

	double angle = rand->getRatio() * 2 * M_PI;
	double x0 = std::cos(angle);
	double x1 = std::sin(angle);

	vector<double> noise{ x0,x1 };
	
	return noise;
}


vector<double> BiasVectorSteppable::white_noise_3d()
{
	//cout << "in the 3d white noise method" << endl;
	RandomNumberGenerator *rand = sim->getRandomNumberGeneratorInstance();

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


void BiasVectorSteppable::randomize_initial_bias()//(CellG *cell)//, bool rnd_inited)
{
	if (!rnd_inited)//we don't want to re-randomize on xml steering
	{
		CellInventory::cellInventoryIterator cInvItr;
		CellG * cell = 0;

		//cerr << "in randomize initial bias" << std::endl;

		for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)
		{
			cell = cellInventoryPtr->getCell(cInvItr);
			if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z == 1)
			{
				vector<double> noise = BiasVectorSteppable::white_noise_2d();
				if (fieldDim.x == 1)
				{
					cell->biasVecX = 0;
					cell->biasVecY = noise[0];
					cell->biasVecZ = noise[1];
				}
				else if (fieldDim.y == 1)
				{
					cell->biasVecX = noise[0];
					cell->biasVecY = 0;
					cell->biasVecZ = noise[1];
				}
				else
				{
					cell->biasVecX = noise[0];
					cell->biasVecY = noise[1];
					cell->biasVecZ = 0;
				}
			}
			else
			{
				vector<double> noise = BiasVectorSteppable::white_noise_3d();
				cell->biasVecX = noise[0];
				cell->biasVecY = noise[1];
				cell->biasVecZ = noise[2];
			}
			//cerr << "in randomize initial bias " << cell->biasVecX << ' ' << cell->biasVecY << ' ' << cell->biasVecZ << ' ' << std::endl;
		}
		rnd_inited = true;
		return;
	}
	else
	{
		return;
	}
}


void BiasVectorSteppable::determine_bias_type(CC3DXMLElement *_xmlData)
{
	if (!_xmlData)
	{
		biasType = WHITE;
	}
	else if (_xmlData->findElement("BiasChange"))
	{
		CC3DXMLElement *_xmlDataBiasChange = _xmlData->getFirstElement("BiasChange");

		std::string biasTypeName = _xmlDataBiasChange->getAttribute("Type");

		transform(biasTypeName.begin(), biasTypeName.end(), biasTypeName.begin(), ::tolower);

		if (biasTypeName == "white")// b = white noise
		{
			biasType = WHITE;
		}
		else if (biasTypeName == "persistent")// b(t+1) = a*b(t) + (1-a)*noise
		{
			biasType = PERSISTENT;
		}
		else if (biasTypeName == "manual")// for changing b in python
		{
			biasType = MANUAL;
		}
		else if (biasTypeName == "custom")// for muExpressions
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
	return;
}


void BiasVectorSteppable::determine_noise_generator()
{
	if (fieldDim.x == 1 || fieldDim.y == 1 || fieldDim.z == 1)
	{
		noiseType = VEC_GEN_WHITE2D;
	}
	else
	{
		noiseType = VEC_GEN_WHITE3D;
	}

	//std::cerr << "noise type" << noiseType << std::endl;
	switch (noiseType)
	{
		case VEC_GEN_WHITE2D:
		{
			noiseFcnPtr = &BiasVectorSteppable::white_noise_2d;
			break;
		}
		case VEC_GEN_WHITE3D:
		{
			noiseFcnPtr = &BiasVectorSteppable::white_noise_3d;
			break;
		}
		default:
		{
			noiseFcnPtr = &BiasVectorSteppable::white_noise_3d;
			break;
		}
	}
	return;
}

void BiasVectorSteppable::determine_field_type()
{
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
	//cerr << "field type " << fieldType << std::endl;
	return;
}

void BiasVectorSteppable::set_white_step_function()
{
	switch (fieldType)
	{
	case CompuCell3D::BiasVectorSteppable::FTYPE3D:
		stepFcnPtr = &BiasVectorSteppable::step_white_3d;
		break;
	case CompuCell3D::BiasVectorSteppable::FTYPE2DX:
		stepFcnPtr = &BiasVectorSteppable::step_white_2d_x;
		break;
	case CompuCell3D::BiasVectorSteppable::FTYPE2DY:
		stepFcnPtr = &BiasVectorSteppable::step_white_2d_y;
		break;
	case CompuCell3D::BiasVectorSteppable::FTYPE2DZ:
		stepFcnPtr = &BiasVectorSteppable::step_white_2d_z;
		break;
	default:
		stepFcnPtr = &BiasVectorSteppable::step_white_2d_x;
		break;
	}
	return;
}

void BiasVectorSteppable::set_persitent_step_function(CC3DXMLElement *_xmlData)
{
	//For now I'll only do by cell type.
	/*<Steppable Type = "BiasVectorSteppable">
	<BiasChange Type = "Persistent" / >
	<BiasChangeParameters CellType = "Cell" Alpha = "0.5">
	< / Steppable>*/

	CC3DXMLElement *_biasXML = _xmlData->getFirstElement("BiasChange");
	CC3DXMLElementList paramVec = _biasXML->getElements("BiasChangeParameters");

	biasPersistParamVec.clear();

	vector<int> typeIdVec;
	vector<BiasPersistParam> biasPersistTemp;

	biasPersistTemp.clear();

	for (int i = 0; i < paramVec.size(); ++i)
	{

		double alpha = paramVec[i]->getAttributeAsDouble("Alpha");
		std::string type = paramVec[i]->getAttribute("CellType");

		BiasPersistParam bParam;

		bParam.persistentAlpha = alpha;
		bParam.typeName = type;

		std::cerr << "automaton=" << automaton << std::endl;
		biasPersistTemp.push_back(bParam);
		typeIdVec.push_back(automaton->getTypeId(type));
	}

	vector<int>::iterator pos = max_element(typeIdVec.begin(), typeIdVec.end());

	int maxTypeId = *pos;
	biasPersistParamVec.clear();
	biasPersistParamVec.assign(maxTypeId+1, BiasPersistParam());

	for (int i = 0; i < biasPersistTemp.size(); ++i)
	{
		biasPersistParamVec[typeIdVec[i]] = biasPersistTemp[i];
		//std::cerr << " in bias persist vec assign " << i << std::endl;
	}

	switch (fieldType)
	{
		case CompuCell3D::BiasVectorSteppable::FTYPE3D:
		{
			//std::cerr << "gen fnc case pers 3d " << std::endl;
			perGenFcnPtr = &BiasVectorSteppable::gen_persistent_bias_3d;
			break;
		}
		case CompuCell3D::BiasVectorSteppable::FTYPE2DX:
		{
			//std::cerr << "gen fnc case pers 2dx " << std::endl;
			perGenFcnPtr = &BiasVectorSteppable::gen_persistent_bias_2d_x;
			break;
		}
		case CompuCell3D::BiasVectorSteppable::FTYPE2DY:
		{
			//std::cerr << "gen fnc case pers 2dy " << std::endl;
			perGenFcnPtr = &BiasVectorSteppable::gen_persistent_bias_2d_y;
			break;
		}
		case CompuCell3D::BiasVectorSteppable::FTYPE2DZ:
		{
			//std::cerr << "gen fnc case pers 2dz " << std::endl;
			perGenFcnPtr = &BiasVectorSteppable::gen_persistent_bias_2d_z;
			break;
		}
		default:
		{
			//std::cerr << "gen fnc case pers def " << std::endl;
			perGenFcnPtr = &BiasVectorSteppable::gen_persistent_bias_2d_x;
			break;
		}
	}

	//std::cerr << "before assign fcn p" << std::endl;
	stepFcnPtr = &BiasVectorSteppable::step_persistent_bias;
	//std::cerr << "after assign fcn p" << std::endl;
	return;

}

void BiasVectorSteppable::set_step_function(CC3DXMLElement *_xmlData)
{
	switch (biasType)
	{
		case CompuCell3D::BiasVectorSteppable::WHITE:
		{
			set_white_step_function();
			break;
		}
		case CompuCell3D::BiasVectorSteppable::PERSISTENT:
		{
			set_persitent_step_function(_xmlData);
			break;
		}
		case CompuCell3D::BiasVectorSteppable::MANUAL:
		{
			set_white_step_function();//PLACE HOLDER
			break;
		}
		case CompuCell3D::BiasVectorSteppable::CUSTOM:
		{	
			set_white_step_function();//PLACE HOLDER
			break;
		}
		default:
		{
			set_white_step_function();
			break;
		}
	}
	return;
}



void BiasVectorSteppable::update(CC3DXMLElement *_xmlData, bool _fullInitFlag)
{
	//PARSE XML IN THIS FUNCTION

	//For more information on XML parser function please see CC3D code or lookup XML utils API

	automaton = potts->getAutomaton();

	if (!automaton) throw CC3DException("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");

	set<unsigned char> cellTypesSet;


	/*
	Example xml:


	<Steppable Type="BiasVectorSteppable">
	<BiasChange Type = "persistent"/>
	<BiasChangeParameters CellType="Cell" Alpha = "0.5"/>
	</Steppable>


	*/

	determine_bias_type(_xmlData);
	//std::cerr << "bias type " << biasType << std::endl;
	
	boundaryStrategy = BoundaryStrategy::getInstance();


	determine_noise_generator();

	determine_field_type();

	set_step_function(_xmlData);
	//std:cerr << "before update return" << std::endl;
	return;
}





std::string BiasVectorSteppable::toString(){

   return "BiasVectorSteppable";

}



std::string BiasVectorSteppable::steerableName(){

   return toString();

}

        

