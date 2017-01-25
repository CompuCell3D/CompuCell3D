
#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <fstream>
#include <sstream>

//#define _DEBUG


#include "CC3DXMLElement.h"
using namespace std;

CC3DXMLElement::CC3DXMLElement(std::string  _name, std::map<std::string,std::string> _attributes,std::string _cdata):
name(_name),attributes(_attributes),cdata(_cdata),defaultIndent(3),elemNameCounterDictPtr(0)
{}	


CC3DXMLElementList::~CC3DXMLElementList(){}
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
	//cerr<<"element="<<this->name<<" comment.size()="<<endl;
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

	//cerr<<"processing element "<<this->name<<endl;
	//cerr<<"cdata "<<this->cdata<<endl;

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
			//cerr<<"i="<<i<<" element="<<children[i]->name<<endl;
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
	
	//cerr<<"element="<<this->name<<" comment.size()="<<endl;

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

	//cerr<<"_commentElemFlag="<<_commentElemFlag<<" commentElemFlag="<<commentElemFlag<<endl;

	//cerr<<"processing element "<<this->name<<endl;
	//cerr<<"cdata "<<this->cdata<<endl;
	string elementName;
	if (!_parentElement.size()){
		//new xml root element - we clear elemNameCounterDict
		//cerr<<"clearing elemNameCounterDict"<<endl;
		elemNameCounterDictPtr->clear();
		elementName=this->name+"Elmnt";
		//cerr<<"root elementName="<<elementName<<endl;

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
			//cerr<<"searching for element "<<this->name<<endl;
			//if (mitr==elemNameCounterDictPtr->end()){
			//	cerr<<"element not found"<<endl;
			//}

			if (mitr!=elemNameCounterDictPtr->end()){
				
				(*elemNameCounterDictPtr)[this->name]+=1;
				//mitr->second+=1;
				ostringstream outStr;
				outStr<<mitr->second;
				elementName=this->name+"Elmnt_"+outStr.str();
				//cerr<<"generated elementName="<<elementName<<endl;
				
			}else{
				elementName=this->name+"Elmnt";
				//cerr<<"new elementName="<<this->name<<endl;
				elemNameCounterDictPtr->insert(make_pair(this->name,0));
				//cerr<<"inserting "<<this->name<<","<<0<<endl;
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
        // _out<<">"/*<<endl*/;	
        // string extraSpace(defaultIndent,' ');
        _out<<",\""<</*leadingSpaces<<extraSpace<<*/this->cdata<<"\""/*<<endl*/;
        _out<<")";
        _out<<endl;
    }else{
        _out<<")"<<endl;	
        
    }		
	
	for(int i=0 ; i < children.size() ; ++i){
		//cerr<<"Writing element "<<i<<" elementName="<<elementName<<endl;			
		children[i]->setElemNameCounterDictPtr(elemNameCounterDictPtr);
		//cerr<<"passing commentElemFlag="<<commentElemFlag<<endl;
		children[i]->writeCC3DXMLElementInPython(_out, elementName,4,commentElemFlag);
		
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CC3DXMLElement * CC3DXMLElement::getFirstElement(std::string  _name, std::map<std::string,std::string> * _attributes){

	//cerr<<"INSIDE getFirstElement"<<endl;
	//if (_attributes){
	//	cerr<<"_attributes->size()="<<_attributes->size()<<endl;
	//}
	//cerr<<"\n\n\n\n\n\n";
	map<string,string>::iterator pos;
	for(int i = 0 ; i< children.size() ; ++i){
		if(children[i]->name==_name){
			if(!_attributes || !_attributes->size())
				return children[i];

			bool attributesMatch=true;
			if(_attributes->size() <= children[i]->attributes.size()){// check attributes when they exist
				for(map<string,string>::iterator mitr = _attributes->begin() ;  mitr != _attributes->end() ; ++mitr){
					//cerr<<"Looking for attribute "<<mitr->first<<endl;
					pos=children[i]->attributes.find(mitr->first);
					if(pos==children[i]->attributes.end() || mitr->second!=pos->second){
						//if there is a mismatch in any attribute (name or value) move to another child element
						attributesMatch=false;
						break;
					}			
				}
				if(attributesMatch)
					return children[i];
				//else
				//	return 0;
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

	//if (findElement(_elementName)){ //such element already exist
	//	return 0;
	//}
	CC3DXMLElement  addedElem(_elementName,std::map<std::string,std::string>(),_cdata);
	//cerr<<" got here"<<endl;

	additionalChildElements.push_back(addedElem);
	//cerr<<" got here 1"<<endl;
	
	addChild(&(*(--additionalChildElements.end())));
	//cerr<<" got here 2"<<endl;
	

	//for (int i = 0 ; i < additionalChildElements.size() ; ++i){
	//	cerr<<"additional element "<<i<<" = "<<additionalChildElements[i].cdata<<endl;
	//}

	//for (int i = 0 ; i < children.size() ; ++i){
	//	cerr<<"additional element "<<i<<" = "<<children[i]->cdata<<endl;
	//}


	return &(*(--additionalChildElements.end()));
	
	//CC3DXMLElement & addedElemRef=*(additionalChildElements.end()-1);
	//addedElemRef.cdata=_cdata;
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
	cerr<<"ADDING CHILDREN of:"<<name<<" child name "<<_child->name<<endl;
#endif
	children.push_back(_child);
#ifdef _DEBUG
	cerr<<"NUMBER OF CHILDREN FOR ELEMENT "<<name<<" is "<<children.size()<<endl;
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
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return BasicString::parseDouble(mitr->second);

}
unsigned int CC3DXMLElement::getAttributeAsUInt(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return BasicString::parseUInteger(mitr->second);

}
int CC3DXMLElement::getAttributeAsInt(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return BasicString::parseInteger(mitr->second);
}

char CC3DXMLElement::getAttributeAsByte(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return BasicString::parseByte(mitr->second);
}

unsigned char CC3DXMLElement::getAttributeAsUByte(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return BasicString::parseUByte(mitr->second);
}


short CC3DXMLElement::getAttributeAsShort(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return BasicString::parseShort(mitr->second);

}
unsigned short CC3DXMLElement::getAttributeAsUShort(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return BasicString::parseUShort(mitr->second);

}
bool CC3DXMLElement::getAttributeAsBool(const std::string key){

	map<std::string,std::string>::iterator mitr=attributes.find(key);
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return BasicString::parseBool(mitr->second);
}



std::string CC3DXMLElement::getAttribute(const std::string key){
	map<std::string,std::string>::iterator mitr=attributes.find(key);
	ASSERT_OR_THROW(string("Element ") + name + "does not have attribute "+key+"!", mitr!=attributes.end());
	return mitr->second;
}

std::string CC3DXMLElement::getData(){
	return cdata;
}
CC3DXMLElementList CC3DXMLElement::getElements(std::string _name){
	CC3DXMLElementList elementList;
	
#ifdef _DEBUG
		cerr<<"getElements for element "<<name<<endl;
#endif //_DEBUG

	if (_name!=""){
		for (CC3DXMLElementList::iterator litr=children.begin() ; litr!=children.end(); ++litr){
			if((*litr)->name==_name)
				elementList.push_back(*litr);
		}
	}else{
#ifdef _DEBUG
		cerr<<"number of children of element "<<name<<" is "<<children.size()<<endl;
		cerr<<"CHILDREN.SIZE()="<<children.size()<<endl;
#endif
		for (CC3DXMLElementList::iterator litr=children.begin() ; litr!=children.end(); ++litr){
			elementList.push_back(*litr);
		}
	}

#ifdef _DEBUG
	cerr<<"LIST COUNT="<<elementList.size()<<endl;
#endif

	return elementList;
}

//CC3DXMLElement* CC3DXMLElement::getFirstElement(std::string _name){
//
//	for (CC3DXMLElementList::iterator litr=children.begin() ; litr!=children.end(); ++litr){
//		if((*litr)->name==_name)
//			return *litr;
//	}
//	return 0;
//}

bool CC3DXMLElement::findElement(const std::string _name, std::map<std::string,std::string> * _attributes){

	return (bool)getFirstElement(_name,_attributes);
	//CC3DXMLElementList elementList=getElements();

	//for (CC3DXMLElementList::iterator litr=children.begin() ; litr!=children.end(); ++litr){
	//	if((*litr)->name==_name)
	//		return true;
	//}
	//return false;

}

unsigned int CC3DXMLElement::getUInt(){
	return BasicString::parseUInteger(cdata);
}

int CC3DXMLElement::getInt(){
	return BasicString::parseInteger(cdata);
}
short CC3DXMLElement::getShort(){
	return BasicString::parseShort(cdata);	
}
unsigned short CC3DXMLElement::getUShort(){
	return BasicString::parseUShort(cdata);	
}
double CC3DXMLElement::getDouble(){
	return BasicString::parseDouble(cdata);	
}

bool CC3DXMLElement::getBool(){
	return BasicString::parseBool(cdata);	
}
std::string CC3DXMLElement::getText(){
	return cdata;	
}

std::string CC3DXMLElement::getName(){
	return name;
}

char CC3DXMLElement::getByte(){
	return BasicString::parseByte(cdata);	
}

unsigned char CC3DXMLElement::getUByte(){
	return BasicString::parseUByte(cdata);	
}
