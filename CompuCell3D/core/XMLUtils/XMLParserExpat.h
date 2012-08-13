#ifndef XMLPARSEREXPAT_H
#define XMLPARSEREXPAT_H

#include <fstream>
#include <expat.h> /* libexpat XML parser */
#include <string>
#include <stack>
#include <XMLUtils/CC3DXMLElement.h>
#include "XMLParserExpatLibDLLSpecifier.h"

class XMLPARSEREXPATLIB_EXPORT XMLParserExpat {
public:
	XMLParserExpat();
	virtual ~XMLParserExpat();
	void setFileName(const std::string &);



	// Tag we're currently working on 
	std::string tag;
	std::string fileName;
	//Number of nested XML tags so far 
	

	int parse();

	int level;
	CC3DXMLElement * rootElement;
	std::stack<CC3DXMLElement * > nodeStack;
	std::list<CC3DXMLElement> elementInventory;
};

	// Called on a start tag, like <Dimensions xDim="10"> 
XMLPARSEREXPATLIB_EXPORT	void handleStartElement(XMLParserExpat *_xmlExpatParser,
		const XML_Char *name, /* Dimensions */
		const XML_Char **atts /* xDim = "10" */
		);


	// Called on an end tag, like </Dimensions> 
XMLPARSEREXPATLIB_EXPORT	void handleEndElement(XMLParserExpat *_xmlExpatParser,
		const XML_Char *name);  //Dimensions 

	// Called on character data found in the file <Steps>10000</Steps>
XMLPARSEREXPATLIB_EXPORT	void handleCharacterData(XMLParserExpat *_xmlExpatParser,
		const XML_Char *data /*10000*/,
		int len);

	// Remove whitespace from start and end of string
XMLPARSEREXPATLIB_EXPORT	std::string squeeze(const std::string &str); 


#endif
