#include <iostream>
#include "CC3DXMLElement.h"


#include "CC3DXMLElementWalker.h"

using namespace std;



CC3DXMLElementWalker::CC3DXMLElementWalker()
{}	


void CC3DXMLElementWalker::iterateCC3DXMLElement(CC3DXMLElement* _element){
	cerr<<"ITERATION ROOT ELEMENT="<<_element->name<<endl;
	CC3DXMLElementList childrenList=_element->getElements();
	if(!childrenList.empty()){
		cerr<<"ELEMENT: "<<_element->name<<" HAS CHILDREN"<<endl;
		for (CC3DXMLElementList::iterator litr=childrenList.begin() ; litr!=childrenList.end();++litr){
			cerr<<"child address="<<&(*litr)<<" childName="<<(*litr)->name<<endl;
			iterateCC3DXMLElement(*litr);
		}
	}else{
		cerr<<"ELEMENT: "<<_element->name<<" CDATA="<<_element->cdata<<endl;
	}

}

