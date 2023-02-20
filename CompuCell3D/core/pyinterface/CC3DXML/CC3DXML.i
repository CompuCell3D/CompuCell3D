
// Module Name
%module CC3DXML

// ************************************************************
// Module Includes
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.


%{
#include <CC3DXMLElement.h>
#include <CC3DXMLElementWalker.h>

#include "STLPyIteratorValueType.h"
#include "STLPyMapIteratorValueType.h"

#define XMLUTILS_EXPORT
// Namespaces
using namespace std;


%}

#define XMLUTILS_EXPORT


// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

//C++ std::list handling
%include "std_list.i"

%template (ListCC3DXMLElement) std::vector<CC3DXMLElement*>;

%include <CC3DXMLElement.h>
%include <CC3DXMLElementWalker.h>

%template (MapStrStr) std::map<std::string,std::string>;
%template (MapIntStr) std::map<int,std::string>;
%template(DoubleMap) std::map<std::string,double>;
//%template (ListCC3DXMLElement) std::list<CC3DXMLElement*>;


%include "STLPyIteratorValueType.h"
%include "STLPyMapIteratorValueType.h"

%template (CC3DXMLElementListIterator) STLPyIteratorValueType<CC3DXMLElementList,CC3DXMLElement*>;

%template (StrStrPair) std::pair<std::string,std::string>;
%template (MapStrStrIterator) STLPyMapIteratorValueType<std::map<std::string,std::string>,std::pair<std::string,std::string> >;


%inline %{
        std::map<std::string,std::string> createStrStrMap(){
           return std::map<std::string,std::string>();
        }
%}

%inline %{
        std::map<int,std::string> createIntStrMap(){
           return std::map<int,std::string>();
        }
%}

%template (ListString) std::list<std::string>;

%inline %{
	
	CC3DXMLElement & derefCC3DXMLElement(CC3DXMLElement * _elem){
		return *_elem;
	}

class STLPyIteratorCC3DXMLElementList
{
public:

    CC3DXMLElementList::iterator current;
    CC3DXMLElementList::iterator begin;
    CC3DXMLElementList::iterator end;


    STLPyIteratorCC3DXMLElementList(CC3DXMLElementList& a)
    {
      initialize(a);
    }

    STLPyIteratorCC3DXMLElementList()
    {
    }

	//CC3DXMLElement *
	
      CC3DXMLElement * getCurrentRef(){
      return const_cast<CC3DXMLElementList::value_type>(*current);
    }
    void initialize(CC3DXMLElementList& a){
        begin = a.begin();
        end = a.end();
    }
    bool isEnd(){return current==end;}
    bool isBegin(){return current==begin;}
    void setToBegin(){current=begin;}

    void previous(){
        if(current != begin){

            --current;
         }

    }

    void next()
    {

        if(current != end){

            ++current;
         }


    }
};
	
	std::list<std::string,std::allocator<std::string > > getListString(){
		std::list<std::string,std::allocator<std::string > > list_string;
		return list_string;
	}

	class ListStringWrapped:public std::list<std::string >{
	
	};
	


	ListStringWrapped getListStringWrapped(){
		ListStringWrapped list_string;
		return list_string;
	}
	

	std::string  getFirstElement(std::list<std::string,std::allocator<std::string > > & _list){
		return *(_list.begin());
	}
	



%}




