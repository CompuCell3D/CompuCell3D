
#include <fstream>
#include <sstream>

//#define _DEBUG


#include "CC3DXMLElement.h"
#include "PublicUtilities/StringUtils.h"
#include <Logger/CC3DLogger.h>

using namespace std;

CC3DXMLElementPtrT::CC3DXMLElementPtrT(CC3DXMLElement* pValue) : pData(pValue)
{
}
CC3DXMLElementPtrT::~CC3DXMLElementPtrT()
    {
        cerr<<"dummy destructor"<<endl;
        // pointer no longer requried
//        delete pData;
    }

    CC3DXMLElement& CC3DXMLElementPtrT::operator* ()
    {
        return *pData;
    }

    CC3DXMLElement* CC3DXMLElementPtrT::operator-> ()
    {
        return pData;
    }





CC3DXMLElement::CC3DXMLElement(std::string  _name, std::map<std::string,std::string> _attributes,std::string _cdata):
name(_name),attributes(_attributes),cdata(_cdata),defaultIndent(3),elemNameCounterDictPtr(0)
{}

CC3DXMLElement::~CC3DXMLElement(){
}


CC3DXMLElementList::~CC3DXMLElementList(){

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DXMLElement::setElemNameCounterDictPtr(map<std::string,int> * _ptr){	
	elemNameCounterDictPtr=_ptr;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DXMLElement::addComment(std::string _comment){
	CC3DXMLElement *commentElement=attachElement("");	
	commentElement->comment=_comment;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DXMLElement::commentOutElement(){
	comment="comel";
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC3DXMLElement * CC3DXMLElementList::getElement(unsigned int _index){
	if (_index < this->size())
		return (*this)[_index];
	else
		return 0;
		

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DXMLElement::saveXML(std::string _fileName){
	ofstream out(_fileName.c_str());
	writeCC3DXMLElement(out);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string CC3DXMLElement::getCC3DXMLElementString(){
	ostringstream out;
	writeCC3DXMLElement(out);
	return out.str();
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CC3DXMLElement::writeCC3DXMLElement(ostream &_out, int _indent){

	

	string leadingSpaces;
	if (_indent){
		leadingSpaces.assign(_indent/*+defaultIndent*/,' ');
	}
	CC3D_Log(CompuCell3D::LOG_TRACE) << "element="<<this->name<<" comment.size()=";
	if (comment.size()){
		if (comment=="comel"){
			_out<<leadingSpaces<<"<!-- ";
		}else{
			if (comment=="newline"){
				_out<<leadingSpaces<<endl;
			}else{
				_out<<leadingSpaces<<"<!-- "<<comment<<" -->"<<endl;
			}
			return;
		}

	}
	CC3D_Log(CompuCell3D::LOG_TRACE) << "processing element "<<this->name;
	CC3D_Log(CompuCell3D::LOG_TRACE) << "cdata "<<this->cdata;

	if (comment!="comel"){
		_out<<leadingSpaces;
	}
	_out<<"<"<<this->name;

	if(attributes.size()){
		for (map<std::string,std::string>::iterator mitr=attributes.begin() ; mitr!=attributes.end() ; ++mitr){
			_out<<" "<<mitr->first<<"=\""<<mitr->second<<"\"";
		}
	}
	

	if(children.size()){
		_out<<">"<<endl;
		if (this->cdata.size()){
			string extraSpace(defaultIndent,' ');
			_out<<leadingSpaces<<extraSpace<<this->cdata<<endl;
		}
		for(int i=0 ; i < children.size() ; ++i){
			CC3D_Log(CompuCell3D::LOG_TRACE) << "i="<<i<<" element="<<children[i]->name;
			int childIndex=(_indent?_indent+defaultIndent:defaultIndent);
			children[i]->writeCC3DXMLElement(_out,childIndex /*_indent+defaultIndent*/);
		}

		_out<<leadingSpaces<<"</"<<this->name<<">";

		if (comment=="comel"){
			_out<<" -->";
		}		
		_out<<endl;

	}else{
		
		if (this->cdata.size()){
			_out<<">"/*<<endl*/;	
			string extraSpace(defaultIndent,' ');
			_out<</*leadingSpaces<<extraSpace<<*/this->cdata/*<<endl*/;
			_out<</*leadingSpaces<<*/"</"<<this->name<<">";
			if (comment=="comel"){
				_out<<" -->";
			}		
			_out<<endl;

		}else{
			_out<<"/>";	
			if (comment=="comel"){
				_out<<" -->";
			}		
			_out<<endl;
			
		}
	}

}

std::string CC3DXMLElement::getXMLAsPython() {
	ostringstream out;
	string parentElement;
	std::map<std::string, int> elemNameCounterDict;
	elemNameCounterDictPtr = &elemNameCounterDict;
	writeCC3DXMLElementInPython(out, parentElement);
	return out.str();


}

void CC3DXMLElement::saveXMLInPython(std::string _fileName){

	ofstream out(_fileName.c_str());
	string parentElement;
	std::map<std::string,int> elemNameCounterDict;
	elemNameCounterDictPtr = & elemNameCounterDict;
	writeCC3DXMLElementInPython(out,parentElement);

}

void CC3DXMLElement::writeCC3DXMLElementInPython(ostream &_out, string _parentElement, int _indent, bool _commentElemFlag){

	string leadingSpaces;
	if (_indent){
		leadingSpaces.assign(_indent/*+defaultIndent*/,' ');
	}
	CC3D_Log(CompuCell3D::LOG_TRACE) << "element="<<this->name<<" comment.size()=";

	bool commentElemFlag=false;

	if (comment=="comel" || _commentElemFlag){
		commentElemFlag=true;
	}

	// since _commentElemFlag=true flag may be passed from parent element (which is commented out) we have to check sperately for thi
	// as Python does not allow block comments
	if (comment.size() || commentElemFlag){
		if (commentElemFlag){
			_out<<leadingSpaces<<"# ";
		}else{
			if (comment=="newline"){
				_out<<leadingSpaces<<endl;
			}else{
				_out<<leadingSpaces<<"# "<<comment<<endl;
			}
			return;
		}

	}
	CC3D_Log(CompuCell3D::LOG_TRACE) << "_commentElemFlag="<<_commentElemFlag<<" commentElemFlag="<<commentElemFlag;
	CC3D_Log(CompuCell3D::LOG_TRACE) << "processing element "<<this->name;
	CC3D_Log(CompuCell3D::LOG_TRACE) << "cdata "<<this->cdata;
	string elementName;
	if (!_parentElement.size()){
		//new xml root element - we clear elemNameCounterDict
		CC3D_Log(CompuCell3D::LOG_TRACE) << "clearing elemNameCounterDict";
		elemNameCounterDictPtr->clear();
		elementName=this->name+"Elmnt";
		CC3D_Log(CompuCell3D::LOG_TRACE) << "root elementName="<<elementName;

		if (!commentElemFlag){
			_out<<leadingSpaces;
		}
		_out<<elementName+"=ElementCC3D("+"\""+this->name+"\"";

		elemNameCounterDictPtr->insert(make_pair(this->name,0));

	}else{
		


		if (children.size()){
			//we only need to generate new element name for those elements that have children
			map<std::string,int>::iterator mitr;
			mitr=elemNameCounterDictPtr->find(this->name);
			CC3D_Log(CompuCell3D::LOG_TRACE) << "searching for element "<<this->name;

			if (mitr!=elemNameCounterDictPtr->end()){
				
				(*elemNameCounterDictPtr)[this->name]+=1;
				ostringstream outStr;
				outStr<<mitr->second;
				elementName=this->name+"Elmnt_"+outStr.str();
				CC3D_Log(CompuCell3D::LOG_TRACE) << "generated elementName="<<elementName;
				
			}else{
				elementName=this->name+"Elmnt";
				CC3D_Log(CompuCell3D::LOG_TRACE) << "new elementName="<<this->name;
				elemNameCounterDictPtr->insert(make_pair(this->name,0));
				CC3D_Log(CompuCell3D::LOG_TRACE) << "inserting "<<this->name<<","<<0;
			}			

			if (!commentElemFlag){
				_out<<leadingSpaces;
			}
			_out<<elementName+"="+_parentElement+".ElementCC3D("+"\""+this->name+"\"";

		}else{

			//elements with no children don't need to store its references in separate variables
			if (!commentElemFlag){
				_out<<leadingSpaces;
			}
			_out<<_parentElement+".ElementCC3D("+"\""+this->name+"\"";
		}
	}
	
	//writing attributes
	if(attributes.size()){
        _out<<",{";
		for (map<std::string,std::string>::iterator mitr=attributes.begin() ; mitr!=attributes.end() ; ++mitr){
            
			_out<<"\""<<mitr->first<<"\":\""<<mitr->second<<"\"";
            if (++mitr!=attributes.end()){
                _out<<",";
            }
            --mitr;
		}
        _out<<"}";
	}
	
    //writing element text
    if (this->cdata.size()){
        if(!attributes.size()){
            _out<<",{}";
        }
        _out<<",\""<</*leadingSpaces<<extraSpace<<*/this->cdata<<"\""/*<<endl*/;
        _out<<")";
        _out<<endl;
    }else{
        _out<<")"<<endl;	
        
    }		
	
	for(int i=0 ; i < children.size() ; ++i){
		CC3D_Log(CompuCell3D::LOG_TRACE) << "Writing element "<<i<<" elementName="<<elementName;
		children[i]->setElemNameCounterDictPtr(elemNameCounterDictPtr);
		CC3D_Log(CompuCell3D::LOG_TRACE) << "passing commentElemFlag="<<commentElemFlag;
		children[i]->writeCC3DXMLElementInPython(_out, elementName,4,commentElemFlag);
		
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC3DXMLElement * CC3DXMLElement::getFirstElement(std::string  _name, std::map<std::string,std::string> * _attributes){
	CC3D_Log(CompuCell3D::LOG_TRACE) << "INSIDE getFirstElement";
	map<string,string>::iterator pos;
	for(int i = 0 ; i< children.size() ; ++i){
		if(children[i]->name==_name){
			if(!_attributes || !_attributes->size())
				return children[i];

			bool attributesMatch=true;
			if(_attributes->size() <= children[i]->attributes.size()){// check attributes when they exist
				for(map<string,string>::iterator mitr = _attributes->begin() ;  mitr != _attributes->end() ; ++mitr){
					CC3D_Log(CompuCell3D::LOG_TRACE) << "Looking for attribute "<<mitr->first;
					pos=children[i]->attributes.find(mitr->first);
					if(pos==children[i]->attributes.end() || mitr->second!=pos->second){
						//if there is a mismatch in any attribute (name or value) move to another child element
						attributesMatch=false;
						break;
					}			
				}
				if(attributesMatch)
					return children[i];
			}
		}
	}
	return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool CC3DXMLElement::checkMatch(std::string  _name, std::map<std::string,std::string> * _attributes){

	if(name!=_name)
		return false;
	else
		if (!_attributes || !_attributes->size())//match ok if names match and there are no attributes to match
			return true;
		else{
			map<string,string>::iterator pos;
			for(map<string,string>::iterator mitr = _attributes->begin() ;  mitr != _attributes->end() ; ++mitr){
				pos=attributes.find(mitr->first);
				if(pos==attributes.end() || mitr->second!=pos->second){
					//if there is a mismatch in any attribute (name or value) return false
					return false;					
				}			
			}
			return true;
		}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DXMLElement::updateElementAttributes(std::map<std::string,std::string> * _attributes){
	if (!_attributes || ! _attributes->size())
		return;

	//update only the attributes that are listed in the _attributes parameter. If the attribute listed in 
	//_attributes does not exist , nothing is done
	map<string,string>::iterator pos;
	for(map<string,string>::iterator mitr = _attributes->begin() ;  mitr != _attributes->end() ; ++mitr){
		pos=attributes.find(mitr->first);
		if(pos!=attributes.end()){
			pos->second=mitr->second;
		}
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CC3DXMLElement::updateElementValue(std::string _cdata){
	cdata=_cdata;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC3DXMLElement * CC3DXMLElement::attachElement(const std::string  _elementName,const std::string  _cdata){
	//client code should check if given element exists and if so take appropriate action/ We can have many elements with same name at same "level" in the XML

	CC3DXMLElement  addedElem(_elementName,std::map<std::string,std::string>(),_cdata);
	additionalChildElements.push_back(addedElem);
	
	addChild(&(*(--additionalChildElements.end())));

	return &(*(--additionalChildElements.end()));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CC3DXMLElement * CC3DXMLElement::attachAttribute(const std::string & _attrName,const std::string & _attrValue){
	if(findAttribute(_attrName))
		return 0;
	pair<std::map<std::string,std::string>::iterator,bool> insertStatus=attributes.insert(make_pair(_attrName,_attrValue));
	if(insertStatus.second)
		return this;
	else 
		return 0;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
long CC3DXMLElement::getPointerAsLong(){
	return (long)this;
}

void CC3DXMLElement::addChild(CC3DXMLElement * _child){
#ifdef _DEBUG
	CC3D_Log(CompuCell3D::LOG_DEBUG) << "ADDING CHILDREN of:"<<name<<" child name "<<_child->name;
#endif
	children.push_back(_child);
#ifdef _DEBUG
	CC3D_Log(CompuCell3D::LOG_DEBUG) << "NUMBER OF CHILDREN FOR ELEMENT "<<name<<" is "<<children.size();
#endif
}

unsigned int CC3DXMLElement::getNumberOfChildren(){
	return children.size();
}


bool CC3DXMLElement::findAttribute(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if(mitr!=attributes.end())
		return true;
	else
		return false;

}
std::map<std::string,std::string> CC3DXMLElement::getAttributes(){
	return attributes;
}



double CC3DXMLElement::getAttributeAsDouble(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!";
	return strToDouble(mitr->second);

}
unsigned int CC3DXMLElement::getAttributeAsUInt(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end();
	return strToUInt(mitr->second);

}
int CC3DXMLElement::getAttributeAsInt(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end();
	return strToInt(mitr->second);
}

char CC3DXMLElement::getAttributeAsByte(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end();
	return strToByte(mitr->second);
}

unsigned char CC3DXMLElement::getAttributeAsUByte(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end();
	return strToUByte(mitr->second);
}


short CC3DXMLElement::getAttributeAsShort(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end();
	return strToShort(mitr->second);

}
unsigned short CC3DXMLElement::getAttributeAsUShort(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end();
	return strToUShort(mitr->second);

}
bool CC3DXMLElement::getAttributeAsBool(const std::string key){

	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end();
	return strToBool(mitr->second);
}



std::string CC3DXMLElement::getAttribute(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	if (mitr==attributes.end()) throw string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end();
	return mitr->second;
}

std::string CC3DXMLElement::getData(){
	return cdata;
}
CC3DXMLElementList CC3DXMLElement::getElements(std::string _name){
	CC3DXMLElementList elementList;
	
#ifdef _DEBUG
		CC3D_Log(CompuCell3D::LOG_DEBUG) << "getElements for element "<<name;
#endif //_DEBUG

	if (_name!=""){
		for (CC3DXMLElementList::iterator litr=children.begin() ; litr!=children.end(); ++litr){
			if((*litr)->name==_name)
				elementList.push_back(*litr);
		}
	}else{
#ifdef _DEBUG
		CC3D_Log(CompuCell3D::LOG_DEBUG) << "number of children of element "<<name<<" is "<<children.size();
		CC3D_Log(CompuCell3D::LOG_DEBUG) << "CHILDREN.SIZE()="<<children.size();
#endif
		for (CC3DXMLElementList::iterator litr=children.begin() ; litr!=children.end(); ++litr){
			elementList.push_back(*litr);
		}
	}

#ifdef _DEBUG
	CC3D_Log(CompuCell3D::LOG_DEBUG) << "LIST COUNT="<<elementList.size();
#endif

	return elementList;
}

bool CC3DXMLElement::findElement(const std::string _name, std::map<std::string,std::string> * _attributes){
	return (bool)getFirstElement(_name,_attributes);
}

unsigned int CC3DXMLElement::getUInt(){
	return (unsigned int)getInt();
}

int CC3DXMLElement::getInt(){
	return stoi(cdata);
}

short CC3DXMLElement::getShort(){
	return (short)getInt();
}

unsigned short CC3DXMLElement::getUShort(){
	return (unsigned short)getInt();
}

double CC3DXMLElement::getDouble(){
	return stod(cdata);
}

bool CC3DXMLElement::getBool(){
	return strToBool(cdata);
}

std::string CC3DXMLElement::getText(){
	return cdata;	
}

std::string CC3DXMLElement::getName(){
	return name;
}

char CC3DXMLElement::getByte(){
	return (char)getUInt();
}

unsigned char CC3DXMLElement::getUByte(){
	return (unsigned char)getUInt();
}
