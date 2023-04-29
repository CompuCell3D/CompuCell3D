#include <Unit.h>
#include <math.h>
//extern struct unit_t  returnedUnit;
#include <iostream>
#include <sstream>
#include <string>
#include <exception>
#include <stdexcept>
#include <CompuCell3D/CC3DExceptions.h>
//#include <unit_calculator_main_lib.h> //declares parseUnit

using namespace std;

Unit::Unit(){			
	//kg=0.;
	//m=0.;		
	//s=0.;
	//A=0.;
	//K=0.;
	//cd=0.;
	//mol=0.;
	////power10multiplier=0;
	//multiplier=1.;
}

Unit::Unit(const struct unit_t &_unit){
	(*this)=(const Unit &)_unit;
}
Unit::Unit(const Unit & _unit){
	(*this)=_unit;	
}


Unit::Unit(const string & _unitStr)
{
	try{
		//unit_t unit=parseUnit(const_cast<char *> (_unitStr.c_str()));
//		(*this)=parseUnit(const_cast<char *> (std::string(_unitStr+"\n").c_str()));
	}catch (logic_error & e){
		throw CompuCell3D::CC3DException(string(e.what()));
	}
	//printUnit(this);
}

//Unit &Unit::operator =(const Unit & _unit){
//	kg=_unit.kg;
//	m=_unit.m;		
//	s=_unit.s;
//	A=_unit.A;
//	K=_unit.K;
//	cd=_unit.cd;
//	mol=_unit.mol;
//	//power10multiplier=_unit.power10multiplier;
//	multiplier=_unit.multiplier;
//	return (*this);
//}


Unit::~Unit(){}


//std::string Unit::toString() const{
//	using namespace std;
//	std::stringstream unitString;
//	//"%.4g*10^%d * kg^%.4g * m^%.4g * s^%.4g * A^%.4g * K^%.4g * mol^%.4g * cd^%.4g\n"
//	unitString<<multiplier/pow(10,(float)((int)log10(multiplier)))<<"*10^"<<(int)log10(multiplier);
//	if(kg)
//		unitString<<" * kg^"<<kg/*<<" * "*/;
//	if(m)
//		unitString<<" * m^"<<m/*<<" * "*/;
//	if(s)
//		unitString<<" * s^"<<s/*<<" * "*/;
//	if(A)
//		unitString<<" * A^"<<A/*<<" * "*/;
//	if(K)
//		unitString<<" * K^"<<K/*<<" * "*/;
//	if(cd)
//		unitString<<" * cd^"<<cd/*<<" * "*/;
//	if(mol)
//		unitString<<" * mol^"<<mol;
//
//	return unitString.str();
//}

