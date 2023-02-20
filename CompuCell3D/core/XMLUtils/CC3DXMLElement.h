#ifndef CC3DXMLELEMENT_H
#define CC3DXMLELEMENT_H

#include <iosfwd> 
#include <iostream>
#include <string>
#include <map>
#include <list>
#include <vector>
#include "XMLUtilsDLLSpecifier.h"
class CC3DXMLElement;

class XMLUTILS_EXPORT CC3DXMLElementPtrT
{
private:
    CC3DXMLElement*    pData; // pointer to person class
public:
    CC3DXMLElementPtrT(CC3DXMLElement* pValue) ;
    ~CC3DXMLElementPtrT();

    CC3DXMLElement& operator* ();
    CC3DXMLElement* operator-> ();

};


class XMLUTILS_EXPORT CC3DXMLElementList:public std::vector<CC3DXMLElement*>{
//may add interface later if necessary
	public:
	typedef std::vector<CC3DXMLElement*>::iterator CC3DXMLElementListIterator_t;
	virtual ~CC3DXMLElementList();
	//added it to make python interfacing a bit easier - should implement separate interface in C++ anyway
	std::vector<CC3DXMLElement*> * getBaseClass(){return (std::vector<CC3DXMLElement*> *)this;}
	virtual CC3DXMLElement * getElement(unsigned int _index);
	//virtual unsigned int getSize();
	//virtual bool empty();


};

class XMLUTILS_EXPORT CC3DXMLElement{
public:

    CC3DXMLElement(std::string  _name, std::map<std::string,std::string> _attributes, std::string _cdata="");
    ~CC3DXMLElement();

    void writeCC3DXMLElement(std::ostream &_out, int _indent=0);
    void writeCC3DXMLElementInPython(std::ostream &_out, std::string _parentElement, int _indent=4,bool _commentElemFlag=false);

    virtual std::string getCC3DXMLElementString();

    void saveXML(std::string _fileName);
    void saveXMLInPython(std::string _fileName);

    std::string getXMLAsPython();




    virtual void addChild(CC3DXMLElement * _child);

    virtual CC3DXMLElement * attachElement(const std::string  _elementName,const std::string  _cdata="");
    virtual CC3DXMLElement * attachAttribute(const std::string & _attrName,const std::string & _attrValue);

    //attribute functions with convenience functions
    virtual bool findAttribute(const std::string key);

    virtual std::map<std::string,std::string> getAttributes();
    virtual std::string getAttribute(const std::string key);
    virtual double getAttributeAsDouble(const std::string key);
    virtual unsigned int getAttributeAsUInt(const std::string key);
    virtual int getAttributeAsInt(const std::string key);
    virtual short getAttributeAsShort(const std::string key);
    virtual unsigned short getAttributeAsUShort(const std::string key);
    virtual bool getAttributeAsBool(const std::string key);
    virtual unsigned char getAttributeAsUByte(const std::string key);
    virtual char getAttributeAsByte(const std::string key);



    virtual std::string getData() ;

    virtual CC3DXMLElementList getElements(std::string _name="");
    //virtual CC3DXMLElement* getFirstElement(std::string _name);
    virtual CC3DXMLElement * getFirstElement(std::string  _name, std::map<std::string,std::string> * _attributes=0);
    virtual bool findElement(const std::string _name, std::map<std::string,std::string> * _attributes=0);
    virtual bool checkMatch(std::string  _name, std::map<std::string,std::string> * _attributes);


    virtual void updateElementAttributes(std::map<std::string,std::string> * _attributes=0);
    virtual void updateElementValue(std::string _cdata);

    virtual unsigned int getNumberOfChildren();

    //Convenience functions to facilitate conversions from text to numbers/other types

    virtual unsigned int getUInt();
    virtual int getInt();
    virtual char getByte();
    virtual unsigned char getUByte();
    virtual short getShort();
    virtual unsigned short getUShort();
    virtual double getDouble();
    virtual bool getBool();
    virtual std::string getText();
    virtual std::string getName();
    virtual long getPointerAsLong();
    void setElemNameCounterDictPtr(std::map<std::string,int> * _ptr);
    virtual void addComment(std::string _comment);
    virtual void commentOutElement();


    std::string name;
	std::string cdata;
	std::map<std::string,std::string> attributes;
	std::list<CC3DXMLElement> additionalChildElements;
	CC3DXMLElementList children;


	std::string comment;

	
private:
	int defaultIndent;
	//std::map<std::string,int> elemNameCounterDict;
	std::map<std::string,int> *elemNameCounterDictPtr;
};


#endif


