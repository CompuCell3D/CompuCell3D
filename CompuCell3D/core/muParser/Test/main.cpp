#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cstdarg>

#include <muParser.h>
#include <limits>
#include <XMLUtils/CC3DXMLElement.h>


class ExpressionEvaluator{
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



class ExpressionEvaluatorDepot{
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
		
        void addVariable(std::string _name);
		void setAlias(std::string _name, std::string _alias);
        void setExpression(std::string _expression);
		unsigned int size(){return eeVec.size();}
		void update(CC3DXMLElement *_xmlData, bool _fullInitFlag);
	private:
        std::string expressionString;
		std::vector<ExpressionEvaluator> eeVec;

};



using namespace mu;
using namespace std;


ExpressionEvaluatorDepot::ExpressionEvaluatorDepot(unsigned int _size){
	eeVec.assign(_size,ExpressionEvaluator());
}

//unsafe but fast
ExpressionEvaluator & ExpressionEvaluatorDepot::operator [](unsigned int _idx){
	return eeVec[_idx];
}

void ExpressionEvaluatorDepot::addVariable(string _name){
	for (unsigned i = 0 ; i < eeVec.size() ; ++i){		
		eeVec[i].addVariable(_name);
	}
}

void ExpressionEvaluatorDepot::setAlias(std::string _name, std::string _alias){
	for (unsigned i = 0 ; i < eeVec.size() ; ++i){		
		eeVec[i].setAlias(_name,_alias);
	}
	
}


void ExpressionEvaluatorDepot::setExpression(std::string _expression){

	expressionString=_expression;
	for (unsigned i = 0 ; i < eeVec.size() ; ++i){
		eeVec[i].setExpression(expressionString);
	}
}

void ExpressionEvaluatorDepot::allocateSize(unsigned int _size){
	eeVec.clear();
	eeVec.assign(_size,ExpressionEvaluator());
}

void ExpressionEvaluatorDepot::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	if(!_xmlData)
		return;
	//<Variable Name="a" Value="0.5"/>
 //   <Variable Name="b" Value="0.7"/>
 //   <BuiltinVariable BuiltinName="CellType" Alias="ct" />
	//<Expression>a+b*ct</Expression>


	//if(!_xmlData){
	//	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1); //use first nearest neighbors for surface calculations as default
	//}else if(_xmlData->getFirstElement("MaxNeighborOrder")){
	//	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("MaxNeighborOrder")->getUInt());
	//}else if (_xmlData->getFirstElement("MaxNeighborDistance")){
	//	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("MaxNeighborDistance")->getDouble());//depth=1.1 - means 1st nearest neighbor
	//}else{
	//	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1); //use first nearest neighbors for surface calculations as default
	//}
	//lmf=boundaryStrategy->getLatticeMultiplicativeFactors();
}



ExpressionEvaluator::ExpressionEvaluator()
// end_mitr(varNameToAddrMap.end())
//end_mitr(varNameToIndexMap.end())
{}

void ExpressionEvaluator::setExpression(std::string _expression){
	try{
		expressionString=_expression;
		p.SetExpr(expressionString);
	}catch(Parser::exception_type &e)
  {
	  cerr << "setExpression Function: "<<e.GetMsg() << endl;
  }

}

void ExpressionEvaluator::setAlias(std::string _name, std::string _alias){

	nameToAliasMap[_name]=_alias;
	map<string,unsigned int>::iterator mitr=varNameToIndexMap.find(_name);
	if(mitr==varNameToIndexMap.end()){	
		string errorStr="Variable "+_name+" undefined. Please define it before defining an alias to it";
		throw mu::ParserError(errorStr.c_str(), 0, "");
	}else{

		map<string,unsigned int>::iterator mitr_local=varNameToIndexMap.find(_alias);
		if(mitr_local!=varNameToIndexMap.end()){
			string errorStr="Proposed alias: "+_alias+" already exists as another variable. Please change alias name";
			throw mu::ParserError(errorStr.c_str(), 0, "");
			
		}
		p.DefineVar(_alias,&varVec[mitr->second]);
		varNameToIndexMap[_alias]=mitr->second;

	}	


}

void ExpressionEvaluator::addVariable(std::string _name){

    // std::map<string,*double>::iterator mitr=varNameToAddrMap.find(_name)
	
    map<string,unsigned int>::iterator mitr=varNameToIndexMap.find(_name);
    //if(mitr==end_mitr){
	if(mitr==varNameToIndexMap.end()){	
        varVec.push_back(0.0);	
        varNameToIndexMap.insert(make_pair(_name,varVec.size()-1));	
        //after adding new variable we have to reinitialize variable locations that pparser has . in practice it means creating new parser and reinitializing it using varNameToIndexMap        
        p=Parser();
        for (mitr=varNameToIndexMap.begin() ; mitr!=varNameToIndexMap.end();++mitr){
//			cerr<<"associating "<<mitr->first<<" with index "<<mitr->second<<endl;
            p.DefineVar(mitr->first, &varVec[mitr->second]);
        }
        if (expressionString.size()){
            p.SetExpr(expressionString);
        }
        
    }
    
}

void ExpressionEvaluator::setVar(std::string _name, double _val){
    map<string,unsigned int>::iterator mitr=varNameToIndexMap.find(_name);

    //if(mitr!=end_mitr){
	if(mitr!=varNameToIndexMap.end()){
        varVec[mitr->second]=_val;
    }
}

//unsafe but fast
void ExpressionEvaluator::setVarDirect(unsigned int _idx, double _val){
    varVec[_idx]=_val;
}

//unsafe but fast
double & ExpressionEvaluator::operator [](unsigned int _idx){
    return varVec[_idx];
}

double ExpressionEvaluator::getVar(std::string _name){
	map<string,unsigned int>::iterator mitr=varNameToIndexMap.find(_name);
    //if(mitr!=end_mitr){
	if(mitr!=varNameToIndexMap.end()){
        return varVec[mitr->second];
    }else{

		return numeric_limits<double>::quiet_NaN(); //NaN
    }
}

//unsafe but fast
double ExpressionEvaluator::getVarDirect(unsigned int _idx){
    return (*this)[_idx];
}


double ExpressionEvaluator::eval(){
	return p.Eval();
	//cerr<<"BEFORE p.Eval"<<endl;	
	//try{

	//	return p.Eval();

	//}catch(Parser::exception_type &e)
 // {
	//  cerr << "eval Function: "<<e.GetMsg() << endl;
 // }
}


// Function callback
double MySqr(double a_fVal) 
{ 
  return a_fVal*a_fVal; 
}

// main program
int main(int argc, char* argv[])
{
  using namespace mu;

   

  //Parser p;
  //vector<double> a(1,10);
  //p.DefineVar("a", &a[0]); 
  //p.SetExpr("a*a");
  //try{

  //cerr<<"eval="<<p.Eval()<<endl;
  //}catch(Parser::exception_type &e)
  //{
  //  cerr << e.GetMsg() << std::endl;
  //}
  //return 0;

  ExpressionEvaluatorDepot eed;
  vector<string> varNames1(2);
  varNames1[0]="a";
  varNames1[1]="b";


  eed.allocateSize(3);  
  eed.addVariables(varNames1.begin(),varNames1.end());

  eed.setExpression("a+b");
  for (unsigned int i =  0 ; i < eed.size() ; ++i){
	  ExpressionEvaluator &ev=eed[i];
	  ev[0]=i*10;
	  ev[1]=i*12;

	  
	  cerr<<"i="<<i<<" x="<<ev.eval()<<endl;

  }





  ExpressionEvaluator ev;
  vector<string> varNames(2);
  varNames[0]="a";
  varNames[1]="b";
  ev.addVariables(varNames.begin(),varNames.end());
  ev.setAlias("a","abc");

  //ev.addVariable("a");
  //ev.addVariable("b");
  ev.setExpression("(a>11) ? (abc+b) : (abc-b)");
  ev[0]=9;
  ev[1]=12;

  const Parser &p = ev.getMuParserObject();
  cerr<<"this is var [0]="<<ev[0]<<endl;
  cerr<<p.Eval()<<endl;
  //cerr<<ev.eval()<<endl;





  //try
  //{
  //  double fVal = 1;
  //  Parser p;
  //  p.DefineVar("a", &fVal); 
  //  p.DefineFun("MySqr", MySqr); 
  //  p.SetExpr("MySqr(a)*_pi+min(10,a)");

  //  for (std::size_t a=0; a<100; ++a)
  //  {
  //    fVal = a;  // Change value of variable a
  //    std::cout << p.Eval() << std::endl;
  //  }
  //}
  //catch (Parser::exception_type &e)
  //{
  //  std::cout << e.GetMsg() << std::endl;
  //}
  //return 0;
}
