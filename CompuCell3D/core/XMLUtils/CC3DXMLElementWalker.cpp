#include <iostream>
#include "CC3DXMLElement.h"
#include<CompuCell3D/CC3DLogger.h>

#include "CC3DXMLElementWalker.h"

using namespace std;



CC3DXMLElementWalker::CC3DXMLElementWalker()
{}	


void CC3DXMLElementWalker::iterateCC3DXMLElement(CC3DXMLElement* _element){
	Log(LOG_DEBUG) << "ITERATION ROOT ELEMENT="<<_element->name;
	CC3DXMLElementList childrenList=_element->getElements();
	if(!childrenList.empty()){
		Log(LOG_DEBUG) << "ELEMENT: "<<_element->name<<" HAS CHILDREN";
		for (CC3DXMLElementList::iterator litr=childrenList.begin() ; litr!=childrenList.end();++litr){
			Log(LOG_DEBUG) << "child address="<<&(*litr)<<" childName="<<(*litr)->name;
			iterateCC3DXMLElement(*litr);
		}
	}else{
		Log(LOG_DEBUG) << "ELEMENT: "<<_element->name<<" CDATA="<<_element->cdata;
	}

}

