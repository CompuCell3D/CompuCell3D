#ifndef rrSymbolH
#define rrSymbolH
#include <string>
#include <ostream>
#include <limits>
#include "rrObject.h"
using std::string;
using std::ostream;

namespace rr
{
class RR_DECLSPEC Symbol : public rrObject
{
    protected:
    public:
        // Set if species also has a rate rule. Use to prevent a dydt being output
        // in the model function if there is a rate rule for it
		bool     			rateRule;
		double              value;
		bool                constant;


	public:
		string              compartmentName;     // Used when symbol is a species
		bool                hasOnlySubstance;     // used when symbol is a species
		string              formula;             // used in case of species defined using initialAmounts;
		string              keyName;             // Used when storing local parameters, keyName is the reaction name
		string              name;

		//Constructors
		Symbol(const string& _name = "", const double& _value = std::numeric_limits<double>::quiet_NaN());
		Symbol(const string& _keyName, const string& _name, const double& _value= std::numeric_limits<double>::quiet_NaN());
		Symbol(const string& _name, const double& _value, const string& _compartmentName);
		Symbol(const string& _name, const double& _value, const string& _compartmentName, const string& _formula);

}; //class rr::Symbol

		ostream& operator<<(ostream& stream, const Symbol& symbol);

}//namespace rr
#endif

//namespace LibRoadRunner.Util
//{
//    public class Symbol
//    {
//        public string compartmentName; // Used when symbol is a species
//        public bool hasOnlySubstance; // used when symbol is a species
//
//        public string formula; // used in case of species defined using initialAmounts;
//        public string keyName; // Used when storing local parameters, keyName is the reaction name
//        public string name;
//
//
//        // Set if species also has a rate rule. Use to prevent a dydt being output
//        // in the model function if there is a rate rule for it
//        public bool rateRule;
//        public double value;
//
//        public Symbol(string name, double value)
//        {
//            this.name = name;
//            this.value = value;
//            rateRule = false;
//        }
//
//        public Symbol(string keyName, string name, double value)
//        {
//            this.keyName = keyName;
//            this.name = name;
//            this.value = value;
//            rateRule = false;
//        }
//
//
//        public Symbol(string name, double value, string compartmentName)
//        {
//            this.name = name;
//            this.value = value;
//            this.compartmentName = compartmentName;
//        }
//
//        public Symbol(string name, double value, string compartmentName, string formula)
//        {
//            this.name = name;
//            this.value = value;
//            this.compartmentName = compartmentName;
//            this.formula = formula;
//        }
//    }
//}
