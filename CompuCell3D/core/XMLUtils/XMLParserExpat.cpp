#include <iostream>
#include <fstream>
#include <cstdio>


#include <XMLUtils/XMLParserExpat.h>
#include <Logger/CC3DLogger.h>

using namespace std;

XMLParserExpat::XMLParserExpat():
level(0),rootElement(0)
{}

XMLParserExpat::~XMLParserExpat(){}

void XMLParserExpat::setFileName(const std::string &_fileName){
	fileName=_fileName;
}



int XMLParserExpat::parse(){

	//XML_Parser parserObject=XML_ParserCreate(NULL);
	XML_Parser parser=XML_ParserCreate(NULL);
	XML_SetUserData(parser,this); //pass this  to our routines

	///* Register routines to get called as the file is parsed */
	XML_SetElementHandler(parser,
		(XML_StartElementHandler)handleStartElement,
		(XML_EndElementHandler)handleEndElement);
	XML_SetCharacterDataHandler(parser,
		(XML_CharacterDataHandler)handleCharacterData);

  ifstream inputStream(fileName.c_str());
  std::string line;
  bool done=false;
  long lineNumber=1;
  while(getline(inputStream,line)) {
	
    if (XML_Parse(parser, line.c_str(), line.size(), done) == XML_STATUS_ERROR) {
		CC3D_Log(LOG_DEBUG) << "ERROR in the XML file: "<<fileName<<" "<<(const char*)XML_ErrorString(XML_GetErrorCode(parser))<<" in line "<< lineNumber;
		const char * error_string=XML_ErrorString(XML_GetErrorCode(parser));
      	return 1;
    }
	++lineNumber;
  }

  XML_ParserFree(parser);

}


void handleStartElement(XMLParserExpat *_xmlExpatParser,const XML_Char *name, const XML_Char **atts ){

	_xmlExpatParser->tag=name;	
#ifdef _DEBUG
	CC3D_Log(LOG_DEBUG) << "Starting tag="<<_xmlExpatParser->tag;
#endif
	map<string,string> atributeDictionary;
	for (int i=0;atts[i];i+=2) {
		atributeDictionary.insert(make_pair(string(atts[i]),string(atts[i+1])));
#ifdef _DEBUG
		CC3D_Log(LOG_DEBUG) << "attrName="<<atts[i]<<" attrValue="<<atts[i+1];
#endif
	}
	CC3DXMLElement element(string(name), atributeDictionary);

	_xmlExpatParser->elementInventory.push_back(element);
	CC3DXMLElement * elementPtr=&_xmlExpatParser->elementInventory.back();

	if (_xmlExpatParser->nodeStack.size()){
			_xmlExpatParser->nodeStack.top()->addChild(elementPtr);
	}else{
        _xmlExpatParser->rootElement = elementPtr;
		
	}
	_xmlExpatParser->nodeStack.push(elementPtr);

	_xmlExpatParser->level++; /* entering a new tag */
}

void handleEndElement(XMLParserExpat *_xmlExpatParser,const XML_Char *name){
	_xmlExpatParser->level--; // leaving a tag
#ifdef _DEBUG
	CC3D_Log(LOG_DEBUG) << "Ending tag: '" << name;
#endif

	_xmlExpatParser->nodeStack.pop();
}

std::string squeeze(const std::string &str)
{
	unsigned int start=0,end=str.size();
	while (start<end && isspace(str[start])) start++;
	while (start<end && isspace(str[end-1])) end--;
	return std::string(str,start,end-start);
}



void handleCharacterData(XMLParserExpat *_xmlExpatParser,const XML_Char *data,int len)
{
	string str=squeeze(string(data,len)); /*<- silly: expat will give you whitespace too */
	if (str.size()>0u)
#ifdef _DEBUG
		CC3D_Log(LOG_DEBUG) << "Character data: '" << str;
#endif
	_xmlExpatParser->nodeStack.top()->cdata=str;
}
