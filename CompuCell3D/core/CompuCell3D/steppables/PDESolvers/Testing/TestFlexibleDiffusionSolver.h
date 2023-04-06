#include "cppunit/TestSuite.h"
#include "cppunit/ui/text/TestRunner.h"
#include "cppunit/TestCaller.h"
#include "cppunit/extensions/HelperMacros.h"

#include "FlexibleDiffusionSolverFE.h"
#include "CompuCell3D/Potts3D/Potts3D.h"
#include "CompuCell3D/Simulator.h"
#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicSmartPointer.h>
#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include "XMLCereal/XMLPullParser.h"
#include "XMLCereal/XMLSerializer.h"
#include <PublicUtilities/StringUtils.h>
#include <xercesc/util/PlatformUtils.hpp>

XERCES_CPP_NAMESPACE_USE;

using namespace std;

namespace CompuCell3D {

    class TestFlexibleDiffusionSolver : public CppUnit::TestFixture {
    private:
        FlexibleDiffusionSolverFE *fds;
        string fileName;
        string consentrationFile;

    public:

        void setUp() {
            fds = new FlexibleDiffusionSolverFE();
            fileName = "diffusion_2D.xml";
            consentrationFile = "diffusion_2D.pulse.txt";

            //FlexibleDiffusionSolverFEParseData *pd 	= new FlexibleDiffusionSolverFEParseData();

            Dim3D dim(57, 57, 3);
            // Dim3D here for concentrationField specifies the size of the array
            // (including border width Array3DBorders<T>::borderWidth which equals 2 in this case)
            //dim.x=55; dim.y=55; dim.z=1;

            float val = 0.0;

            fds->fieldDim.x = 55;
            fds->fieldDim.y = 55;
            fds->fieldDim.z = 1;
            fds->concentrationFieldVector.push_back(new Array3DBordersField3DAdapter<float>(dim, val));
        }

        void tearDown() {
            delete fds;
        }

        void test_readConcentrationField() {
            fds->readConcentrationField(consentrationFile, fds->concentrationFieldVector[0]);

            CPPUNIT_ASSERT(2000 == fds->concentrationFieldVector[0]->get(Point3D(0, 0, 0)));
        }

        void test_initializeConcentration() {
            //readXMLSteppable(); // initialize fds->fdspd
            //fds->fdspdPtr = &fds->fdspd;
            //fds->initializeConcentration();

            //CPPUNIT_ASSERT( 2000 == fds->concentrationFieldVector[0]->get(Point3D(0,0,0)) );
        }

        // From FlexibleDiffusionSolverFE

        void readXMLSteppable() {
/*
	    XMLPlatformUtils::Initialize();

	   // XMLPullParser::createInstance() is a static method
		BasicSmartPointer<XMLPullParser> parser = XMLPullParser::createInstance();

		parser->initParse(fileName);
		parser->match(XMLEventTypes::START_DOCUMENT, -XMLEventTypes::TEXT);
		parser->assertName("CompuCell3D"); // Parse
		parser->match(XMLEventTypes::START_ELEMENT);

		// Simulator::readXML()
		parser->skip(XMLEventTypes::TEXT); // Very sensitive to XML tree structure

		while (parser->check(XMLEventTypes::START_ELEMENT))
		{

			// It will not parse other elements until it parses Potts, and Plugin
			// End elements (parser->match(XMLEventTypes::END_ELEMENT, -XMLEventTypes::TEXT)) and end document (parser->match(XMLEventTypes::END_DOCUMENT) for some reason do not work

			if (parser->getName() != "Steppable")
				parser->next(XMLEventTypes::START_ELEMENT);
			else
			{
				parser->match(XMLEventTypes::START_ELEMENT);
				fds->readXML(* parser);
				//parser->match(XMLEventTypes::END_ELEMENT, -XMLEventTypes::TEXT);
			}

		}
*/
        }

        CPPUNIT_TEST_SUITE( TestFlexibleDiffusionSolver );
        CPPUNIT_TEST( test_readConcentrationField );
        CPPUNIT_TEST( test_initializeConcentration );

        CPPUNIT_TEST_SUITE_END();

    };

}

int runTestFlexibleDiffusionSolver() {

    CPPUNIT_TEST_SUITE_REGISTRATION(CompuCell3D::TestFlexibleDiffusionSolver);

    CppUnit::TextUi::TestRunner runner;
    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
    runner.addTest(registry.makeTest());
    if (runner.run())
        return 0;
    else
        return 1;

};

/*
   void testInvalidTitle() throw (std::exception)
   {

   }

   void testAlwaysFails() {
      CPPUNIT_FAIL( "Expected failure" );
   }

//   CPPUNIT_TEST_EXCEPTION( testInvalidTitle, std::exception );
//   CPPUNIT_TEST_FAIL( testAlwaysFails );

*/


