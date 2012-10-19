#ifndef EXPRESSIONEVALUATOR_H
#define EXPRESSIONEVALUATOR_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cstdarg>

#include <muParser/muParser.h>
#include <limits>
#include <XMLUtils/CC3DXMLElement.h>
#include "ExpressionEvaluatorDLLSpecifier.h"


class EXPRESSIONEVALUATOR_EXPORT ExpressionEvaluator{
    public:
        ExpressionEvaluator();
		
        template <class ForwardIterator>
		void addVariables(ForwardIterator first, ForwardIterator last){			  
			  if (first==last) return ;
			  for (ForwardIterator fitr=first; fitr!=last ; ++fitr){				  
				  this->addVariable(*fitr);
			  }

		}
        void addVariable(std::string _name);
		void setAlias(std::string _name, std::string _alias);
        void setExpression(std::string _expression);
        void setVar(std::string _name, double _val);
        void setVarDirect(unsigned int _idx, double _val);
        
        double getVar(std::string _name);
        double getVarDirect(unsigned int _idx);

        double & operator [](unsigned int _idx);
        double eval();
		const mu::Parser & getMuParserObject(){return p;}
		std::map<std::string,unsigned int> getVarNameToIndexMap(){return varNameToIndexMap;};
		unsigned int getNumberOfVars(){return varVec.size();}

    private:
        // std::map<std::string,*double> varNameToAddrMap;
        std::map<std::string,unsigned int> varNameToIndexMap;
        std::vector<double> varVec;
		std::map<std::string,std::string> nameToAliasMap;
        // std::map<string,*double>::iterator end_mitr;
		//std::map<std::string,unsigned int>::iterator end_mitr;
        
		mu::Parser p;
        std::string expressionString;
        
};

class EXPRESSIONEVALUATOR_EXPORT ExpressionEvaluatorDepotParseData{
public:

	std::vector<std::string> varNameVec;
	std::vector<double> varValueVec;
	std::vector<std::pair<std::string,std::string> > builtinVariableAliasPairVec;
	std::string expression;	
};

class EXPRESSIONEVALUATOR_EXPORT ExpressionEvaluatorDepot{
	public:

		ExpressionEvaluatorDepot(unsigned int _size=0);

		void allocateSize(unsigned int _size=0);

		ExpressionEvaluator & operator [](unsigned int _idx);		

        template <class ForwardIterator>
		void addVariables(ForwardIterator first, ForwardIterator last){			  
				
			  if (first==last) return ;

			for (unsigned i = 0 ; i < eeVec.size() ; ++i){
			  for (ForwardIterator fitr=first; fitr!=last ; ++fitr){				  
				  eeVec[i].addVariable(*fitr);
			  }				
			}


		}

        template <class ForwardIterator>

		void setVariables(ForwardIterator first, ForwardIterator last,unsigned int offset=0){			  
				
			  if (first==last) return ;
			  
			for (unsigned i = 0 ; i < eeVec.size() ; ++i){

              unsigned int idx=0;
			  for (ForwardIterator fitr=first; fitr!=last ; ++fitr){				  
				  eeVec[i][offset+idx]=*fitr;
					++idx;
			  }				
			}


		}


        void addVariable(std::string _name);
		
		void setAlias(std::string _name, std::string _alias);
        void setExpression(std::string _expression);
		unsigned int size(){return eeVec.size();}
		unsigned int getNumberOfVars(){return eeVec.size() ? eeVec[0].getNumberOfVars(): 0 ;}
		void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);

		void getParseData(CC3DXMLElement *_xmlData);
		void initializeUsingParseData();
		std::string getExpressionString(){return expressionString;}


	private:
        std::string expressionString;
		std::vector<ExpressionEvaluator> eeVec;
		ExpressionEvaluatorDepotParseData parseData;

};



#endif
