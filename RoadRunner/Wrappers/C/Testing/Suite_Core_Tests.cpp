#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "Poco/File.h"
#include "Poco/DOM/ProcessingInstruction.h"
#include "Poco/DOM/DOMParser.h"
#include "Poco/DOM/AutoPtr.h"
#include "Poco/DOM/Document.h"
#include "Poco/DOM/NodeIterator.h"
#include "Poco/DOM/NodeFilter.h"
#include "Poco/DOM/DOMWriter.h"
#include "Poco/DOM/TreeWalker.h"
#include "Poco/SAX/InputSource.h"
#include "Poco/MD5Engine.h"
#include "UnitTest++.h"
#include "rrc_api.h"
#include "rrUtils.h"
#include "rrException.h"

using namespace UnitTest;
using namespace std;
using namespace rr;
using namespace rrc;
using namespace Poco;
using namespace Poco::XML;

//using namespace Poco::XML::NodeFilter;


extern string   gTempFolder;
extern string   gTestDataFolder;
extern bool		gDebug;

string getListOfReactionsText(const string& fName);

SUITE(CORE_TESTS)
{
	TEST(COMPILER)
	{
    	RRHandle aRR 		  		= createRRInstanceE(gTempFolder.c_str());
        //Copy test model sources to temp folder, and compile there

		Poco::File headerFile(JoinPath(gTestDataFolder, "ModelSourceTest.h"));
		Poco::File sourceFile(JoinPath(gTestDataFolder, "ModelSourceTest.c"));
        headerFile.copyTo(gTempFolder);
        sourceFile.copyTo(gTempFolder);

        string testSource = JoinPath(gTempFolder, "ModelSourceTest.c");
		compileSource(aRR, testSource.c_str());
        freeRRInstance(aRR);
	}

    TEST(RELOADING_MODEL_MODEL_RECOMPILIATION)
    {
    	RRHandle aRR 		  		= createRRInstanceE(gTempFolder.c_str());
		string TestModelFileName 	= JoinPath(gTestDataFolder, "Test_1.xml");
		CHECK(FileExists(TestModelFileName));

		CHECK(loadSBMLFromFileE(aRR, TestModelFileName.c_str(), true));

        //Load the same model again, but do not recompile the model DLL..
		CHECK(loadSBMLFromFileE(aRR, TestModelFileName.c_str(), true));
        freeRRInstance(aRR);
    }

    TEST(RELOADING_MODEL_NO_MODEL_RECOMPILIATION)
    {
    	RRHandle aRR 		  		= createRRInstanceE(gTempFolder.c_str());
		string TestModelFileName 	= JoinPath(gTestDataFolder, "Test_1.xml");
		CHECK(FileExists(TestModelFileName));

		CHECK(loadSBMLFromFileE(aRR, TestModelFileName.c_str(), true));

        //Load the same model again, but do not recompile the model DLL..
		CHECK(loadSBMLFromFileE(aRR, TestModelFileName.c_str(), false));
        freeRRInstance(aRR);
    }

    TEST(LOADING_MODEL_MULTIPLE_INSTANCES)
    {
    	RRHandle aRR1 		  		= createRRInstanceE(gTempFolder.c_str());
    	RRHandle aRR2 		  		= createRRInstanceE(gTempFolder.c_str());
		string TestModelFileName 	= JoinPath(gTestDataFolder, "Test_1.xml");

		CHECK(loadSBMLFromFileE(aRR1, TestModelFileName.c_str(), true));
		CHECK(loadSBMLFromFileE(aRR2, TestModelFileName.c_str(), true));

        //Load the same model again, but do not recompile the model DLL..
        CHECK(loadSBMLFromFileE(aRR1, TestModelFileName.c_str(), false));
        CHECK(loadSBMLFromFileE(aRR2, TestModelFileName.c_str(), false));

        freeRRInstance(aRR1);
        freeRRInstance(aRR2);
    }

    TEST(PARSING_MODEL_XML)
    {
    	string modelXML = getListOfReactionsText(JoinPath(gTestDataFolder, "Test_1.xml").c_str());
        CHECK(modelXML.size() > 0);
    }

    TEST(GENERATING_MODEL_HASH)
    {
        string content = getListOfReactionsText(JoinPath(gTestDataFolder, "Test_1.xml"));
        MD5Engine md5;
        md5.update(content);
		string digestString(Poco::DigestEngine::digestToHex(md5.digest()));
//        clog<<digestString;
		CHECK_EQUAL("8b0f11b35815fd421d32ab98750576ef", digestString);
    }

    TEST(LOAD_MODEL_FROM_STRING)
    {
        string xml = GetFileContent(JoinPath(gTestDataFolder, "Test_1.xml"));
    	RRHandle aRR1 		  		= createRRInstanceE(gTempFolder.c_str());
    	RRHandle aRR2 		  		= createRRInstanceE(gTempFolder.c_str());
		CHECK(loadSBML(aRR1, xml.c_str()));
		CHECK(loadSBMLE(aRR2, xml.c_str(), true));

        //Load the same model again, but do not recompile the model DLL..
        CHECK(loadSBMLE(aRR1, xml.c_str(), false));
        CHECK(loadSBMLE(aRR2, xml.c_str(), false));

        freeRRInstance(aRR1);
        freeRRInstance(aRR2);

    }

}

string getListOfReactionsText(const string& fName)
{
		ifstream in(JoinPath(gTestDataFolder, "Test_1.xml").c_str());
        InputSource src(in);
        DOMParser parser;
        AutoPtr<Document> pDoc = parser.parse(&src);
		TreeWalker it(pDoc, Poco::XML::NodeFilter::SHOW_ALL);

        Node* pNode = it.nextNode();
		string result;
        while(pNode)
        {
			clog<<pNode->nodeName()<<endl;
            string nodeID = "listOfReactions";
        	if(ToUpper(pNode->nodeName()) == ToUpper(nodeID))
            {
            	DOMWriter aWriter;
                stringstream xml;
                aWriter.writeNode(xml, pNode);
                result = xml.str();
                break;
            }
            pNode = it.nextNode();
        }

		result.erase( std::remove_if( result.begin(), result.end(), ::isspace ), result.end() );
		return result;
}





