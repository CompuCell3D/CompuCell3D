#include "cppunit/TestCase.h"
#include "cppunit/TestSuite.h"
#include "cppunit/ui/text/TestRunner.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"

#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>

#include "mainCC3D.h"
#include "Simulator.h"
#include <CompuCell3D/Field3D/Array3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include "CompuCell3D/steppables/PDESolvers/FlexibleDiffusionSolverFE.h"
#include "CompuCell3D/steppables/PDESolvers/FlexibleDiffusionSolverADE.h"
#include "CompuCell3D/ClassRegistry.h"
#include <BasicUtils/BasicSmartPointer.h>
#include <XMLCereal/XMLPullParser.h>
#include <XercesUtils/XercesStr.h>
#include <xercesc/util/PlatformUtils.hpp>
XERCES_CPP_NAMESPACE_USE;

//#define array2D_t vector<vector<float> >;

// This test repeats functionality of the class CC3DTransaction (CompuCellPlayer/simthread/mainCC3D.h) and main() function CompuCell3D/main.cpp.

using namespace std;

namespace CompuCell3D
{
class FlexibleDiffusionSolverFE;
class FlexibleDiffusionSolverADE;

class TestMainCC3D : public CppUnit::TestFixture
{
private:
	Simulator *sim;
	vector<vector<vector<float> > > newConcentration;
	vector<vector<vector<float> > > oldConcentration;
	std::vector<Array3DBordersField3DAdapter<float> * > concentration;
	string simulationFileName;

public:

   void setUp()
   {
		// Create Simulator
		sim = new Simulator();

	   // Does it work on Windows?
	   setenv("COMPUCELL3D_STEPPABLE_PATH", "/home/dexity/CompuCellBin/lib/CompuCell3DSteppables", 1);
	   setenv("COMPUCELL3D_PLUGIN_PATH", "/home/dexity/CompuCellBin/lib/CompuCell3DPlugins", 1);
   }

   void tearDown()
   {
	   delete sim;
   }

   void test_FlexibleDiffusionSolverFE_diffuse2D()
   {
	   initTestConcentration(15, 15, 1);
	   initSimulation("diffusion_2D_FE.xml");
	   concentration = ((FlexibleDiffusionSolverFE *)sim->getClassRegistry()->getStepper("FlexibleDiffusionSolverFE"))->getConcentrationFieldVector();
	   runSimulation(0.01, 0.0001); 	// Specific
	   sim->finish();
   }

   void test_FlexibleDiffusionSolverFE_diffuse3D()
   {
	   initTestConcentration(15, 15, 15);
	   initSimulation("diffusion_3D_FE.xml");
	   concentration = ((FlexibleDiffusionSolverFE *)sim->getClassRegistry()->getStepper("FlexibleDiffusionSolverFE"))->getConcentrationFieldVector();
	   runSimulation(0.01, 0.0001);
	   sim->finish();
   }

   void test_FlexibleDiffusionSolverADE_diffuse2D()
   {
	   initTestConcentration(15, 15, 1);
	   initSimulation("diffusion_2D_ADE.xml");
	   concentration = ((FlexibleDiffusionSolverADE *)sim->getClassRegistry()->getStepper("FlexibleDiffusionSolverADE"))->getConcentrationFieldVector();
	   runSimulation(0.01, 1);
	   sim->finish();
   }

   void test_FlexibleDiffusionSolverADE_diffuse3D()
   {
	   initTestConcentration(15, 15, 15);
	   initSimulation("diffusion_3D_ADE.xml");
	   concentration = ((FlexibleDiffusionSolverADE *)sim->getClassRegistry()->getStepper("FlexibleDiffusionSolverADE"))->getConcentrationFieldVector();
	   runSimulation(0.01, 1);
	   sim->finish();
   }


   void runSimulation(float diffConst, float accuracy)
   {
		cerr << "fieldDim:\t" << sim->getPotts()->getCellFieldG()->getDim().x << "x" << sim->getPotts()->getCellFieldG()->getDim().y << "x" << sim->getPotts()->getCellFieldG()->getDim().z << endl;

		for (unsigned int i = 1; i <= 40; i++)//(i <= sim.getNumSteps())
		{
			sim->step(i);
			stepConcentration(i, diffConst);

			if (i==40)
			{
				dumpLatticeConcentration(*sim);
				dumpTestConcetration(newConcentration);

				for (int i = 0; i < newConcentration.size(); i++)
					for (int j = 0; j < newConcentration[0].size(); j++)
						for (int k = 0; k < newConcentration[0][0].size(); k++)
						{
							//cerr << "TestC[" << i << "][" << j << "][" << k << "] =\t " << newConcentration[i][j][k] << "\t| " << concentration[0]->get(Point3D(i,j,k)) << "\t-- Actual" << endl;
							CPPUNIT_ASSERT_DOUBLES_EQUAL(newConcentration[i][j][k], concentration[0]->get(Point3D(i,j,k)), accuracy);
						}
			}
		}

   }

   void initSimulation(string fileName)
   {
	   simulationFileName = fileName;

	   // Load libraries directly. Make sure that you installed the libraries (make install)
	   // before you load them

		char *steppablePath 	= getenv("COMPUCELL3D_STEPPABLE_PATH");
		char *pluginPath 		= getenv("COMPUCELL3D_PLUGIN_PATH");

		cerr << "steppablePath = " 	<< steppablePath << endl;
		cerr << "pluginPath = " 	<< pluginPath << endl;

		if (steppablePath) 	Simulator::steppableManager.loadLibraries(steppablePath);
		if (pluginPath) 	Simulator::pluginManager.loadLibraries(pluginPath);

		BasicPluginManager<Plugin>::infos_t *infos = &Simulator::pluginManager.getPluginInfos();

		if (!infos->empty())
		{
			cerr << "Found the following plugins:" << endl;
			BasicPluginManager<Plugin>::infos_t::iterator it;
			for (it = infos->begin(); it != infos->end(); it++)
			cerr << "  " << *(*it) << endl;
			cerr << endl;
		}

		// Initialize Xerces
		XMLPlatformUtils::Initialize();

		// Create parser
		BasicSmartPointer<XMLPullParser> parser = XMLPullParser::createInstance();

		try
		{
/*
			parser->initParse(simulationFileName);
			parser->match(XMLEventTypes::START_DOCUMENT, -XMLEventTypes::TEXT);

			// Parse
			parser->assertName("CompuCell3D");
			parser->match(XMLEventTypes::START_ELEMENT);
			sim->readXML(*parser);
			parser->match(XMLEventTypes::END_ELEMENT, -XMLEventTypes::TEXT);

			// End
			parser->match(XMLEventTypes::END_DOCUMENT);
*/
			sim->initializeCC3D();
		}
		catch (BasicException e)
		{
			throw BasicException("While parsing configuration!", parser->getLocation(), e);
		}


		string moduleName 	= sim->ps.steppableParseDataVector[0]->moduleName;

		//CPPUNIT_ASSERT_EQUAL (string("FlexibleDiffusionSolverFE"), sim->ps.steppableParseDataVector[0]->moduleName); // Should give FlexibleDiffusionSolverFE

		//sim.extraInit();// NOT USED HERE. Registers Serializable if serializeFlag == true

		cerr << endl;
		// Run simulation
		sim->start();

   }

   void dumpLatticeConcentration(Simulator &sim)
   {
		for (int i=0; i< sim.getPotts()->getCellFieldG()->getDim().x; i++) //latticeSizeX = sim.getPotts()->getCellFieldG()->getDim().x
		{
			for (int j=0; j< sim.getPotts()->getCellFieldG()->getDim().y; j++)
				cerr << concentration[0]->get(Point3D(i,j,0)) << "\t"; // Plane z=0

			cerr << endl;
		}
   }

   void initTestConcentration(int dimX, int dimY, int dimZ)
   {
	   // No periodic
	   vector<float> tempCon;
	   vector<vector<float> > tempVecCon;

	   // Init concentration
	   for (int i = 0; i < dimX; i++)
	   {
		   for (int j = 0; j < dimY; j++)
		   {
			   for (int k = 0; k < dimZ; k++)
				   tempCon.push_back((float)0);

			   tempVecCon.push_back(tempCon);
		   }

		   newConcentration.push_back(tempVecCon);
	   }

	   newConcentration[0][0][0] = 2000;
   }

   void stepConcentration(int step, float diffConst)
   {
	   oldConcentration = newConcentration;
	   // Periodic boundary conditions. They are simpler :)
	   int dimX = newConcentration.size();
	   int dimY = newConcentration[0].size();
	   int dimZ = newConcentration[0][0].size();

	   if (diffConst == (float)0)
		   diffConst = 0.01;

	   for (int i=0; i < dimX; i++)
		   for (int j=0; j < dimY; j++)
			   for (int k=0; k < dimZ; k++)
			   {
				   newConcentration[i][j][k] = diffConst*(
						   oldConcentration[pb(i+1, dimX)][j][k] + oldConcentration[i][pb(j+1, dimY)][k] +
						   oldConcentration[pb(i-1, dimX)][j][k] + oldConcentration[i][pb(j-1, dimY)][k] +
						   oldConcentration[i][j][k]*(1/diffConst - 4) );

				   if (dimZ > 1)
				   {
					   newConcentration[i][j][k] += diffConst*(
						   oldConcentration[i][j][pb(k+1, dimZ)] + oldConcentration[i][j][pb(k-1, dimZ)] -
						   2*oldConcentration[i][j][k]);
				   }
			   }

/*
	   cerr << "C[0][0]" << validConcentration[0][0] << endl;
	   cerr << "C[1][0]" << validConcentration[1][0] << endl;
	   cerr << "C[0][1]" << validConcentration[0][1] << endl;
	   cerr << "C[-1][0]" << validConcentration[pb(-1, dimY)][0] << endl;
	   cerr << "C[0][-1]" << validConcentration[0][pb(-1, dimX)] << endl;
*/
   }

   int pb (int index, int dim)
   {
	   int temp = 0;

	   if (index < 0)
		   temp =  index + ((abs(index)-1)/dim+1)*dim;//temp =  dim + index - 1;
	   else if (index >= dim)
		   temp = index%dim;
	   else
		   temp = index;

	   return temp;
   }

   void dumpTestConcetration(vector<vector<vector<float> > > &con)
   {
	   ofstream file;

	   //file.open("diffusion2D_output.txt");

	   // Dumps values of concentration in the plane z=0

	   cerr << endl << "Dumping Valid Concentration\n";
	   for (int i = 0; i < con.size(); i++)
	   {
		   for (int j = 0; j < con[0].size(); j++)
		   {
			   cerr << con[i][j][0] << "\t"; // Plane z=0
			   //file << con[i][j][0] << "\t";
		   }

		   cerr << endl;
		   //file << endl;
	   }
	   //file.close();
   }

   CPPUNIT_TEST_SUITE( TestMainCC3D );
   CPPUNIT_TEST( test_FlexibleDiffusionSolverFE_diffuse2D );
   //CPPUNIT_TEST( test_FlexibleDiffusionSolverFE_diffuse3D );
   //CPPUNIT_TEST( test_FlexibleDiffusionSolverADE_diffuse2D );
   //CPPUNIT_TEST( test_FlexibleDiffusionSolverADE_diffuse3D );
   CPPUNIT_TEST_SUITE_END();

};

}

// USEFUL VARIABLES:
// int latticeSizeX 				= sim.getPotts()->getCellFieldG()->getDim().x;
// string ConcentrationFieldName 	= ((FlexibleDiffusionSolverFE *)sim.getClassRegistry()->getStepper("FlexibleDiffusionSolverFE"))->getConcentrationFieldNameVector()[0];
// float diffConst 					= ((FlexibleDiffusionSolverFE *)sim.getClassRegistry()->getStepper("FlexibleDiffusionSolverFE"))->fdspdPtr->diffSecrFieldTuppleVec[0].diffData.decayConst;
// float concentration_x_y_z 		= concentration[0]->get(Point3D(x,y,z))

int runTestMainCC3D()
{

   CPPUNIT_TEST_SUITE_REGISTRATION( CompuCell3D::TestMainCC3D );

   CppUnit::TextUi::TestRunner runner;
   CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
   runner.addTest( registry.makeTest() );
   if ( runner.run() )
      return 0;
   else
      return 1;

};
