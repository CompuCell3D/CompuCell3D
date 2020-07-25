
// Module Name
%module CC3DXML
// %module ("threads"=1) CC3DXML

// %ignore SwigPyIterator; 
// #define SwigPyIterator CC3DXML_SwigPyIterator
// %{
// #define SwigPyIterator CC3DXML_SwigPyIterator
// %}
// %rename (CC3DXML_SwigPyIterator) SwigPyIterator;
// %include "pyiterators.swg"

// ************************************************************
// Module Includes
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.


%{
#include <CC3DXMLElement.h>
#include <CC3DXMLElementWalker.h>

#include <pyinterface/CompuCellPython/STLPyIteratorValueType.h>
#include <pyinterface/CompuCellPython/STLPyMapIteratorValueType.h>

#define XMLUTILS_EXPORT
// Namespaces
using namespace std;


%}

#define XMLUTILS_EXPORT
// %ignore SwigPyIterator;
// %rename (SwigPyIteratorCC3DXML) SwigPyIterator;



// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

//C++ std::list handling
%include "std_list.i"

%include <CC3DXMLElement.h>
%include <CC3DXMLElementWalker.h>

%template (MapStrStr) std::map<std::string,std::string>;
%template (MapIntStr) std::map<int,std::string>;
%template(DoubleMap) std::map<std::string,double>;
//%template (ListCC3DXMLElement) std::list<CC3DXMLElement*>;
%template (ListCC3DXMLElement) std::vector<CC3DXMLElement*>;

%include <pyinterface/CompuCellPython/STLPyIteratorValueType.h>
%include <pyinterface/CompuCellPython/STLPyMapIteratorValueType.h>

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
		list_string.push_back("str1");
		list_string.push_back("str2");
		list_string.push_back("str3");
		return list_string;
	}

	class ListStringWrapped:public std::list<std::string >{
	
	};
	


	ListStringWrapped getListStringWrapped(){
		ListStringWrapped list_string;
		list_string.push_back("str1");
		list_string.push_back("str2");
		list_string.push_back("str3");
		return list_string;
	}
	

	std::string  getFirstElement(std::list<std::string,std::allocator<std::string > > & _list){
		return *(_list.begin());
	}
	
	class Try{
		public:
			int a;
			std::list<std::string> listString;
	};

	Try getTry(){
		Try t;
		t.a=10;
		t.listString.push_back("trtrtrtrtrtr1");
		t.listString.push_back("trtrtrtrtrtr2");
		t.listString.push_back("trtrtrtrtrtr3");
		return t;
	}


%}




