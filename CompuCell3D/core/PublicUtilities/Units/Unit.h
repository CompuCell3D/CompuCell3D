#ifndef UNIT_H
#define UNIT_H

# include <PublicUtilities/Units/unit_calculator.h>
# include <iostream>
#include <cmath>
#include <Logger/CC3DLogger.h>

class Unit{
	//:public unit_t{

public:
	Unit();
	Unit(const unit_t &_unit);
	Unit(const Unit & _unit);
	Unit(const std::string & _unitStr);
	//Unit & operator =(const Unit & _unit);

	~Unit();

	//std::string toString() const;
	//inline bool operator==(const Unit & _rhs);
	//inline bool operator!=(const Unit & _rhs);

	//inline Unit & operator*=(const Unit & _rhs);
	//inline Unit & operator/=(const Unit & _rhs);

	//inline Unit & operator*=(double _rhs);

	//inline Unit & operator/=(double _rhs);
	//inline double multiplicativeFactor() const;

};

//inline bool Unit::operator==(const Unit & _rhs){
//	return (kg==_rhs.kg && m==_rhs.m && s==_rhs.s && A==_rhs.A && K==_rhs.K && cd==_rhs.cd && mol == _rhs.mol);
//}
//inline bool Unit::operator!=(const Unit & _rhs){
//	return (kg!=_rhs.kg || m!=_rhs.m || s!=_rhs.s || A!=_rhs.A || K!=_rhs.K || cd!=_rhs.cd || mol != _rhs.mol);
//}
//
//inline Unit & Unit::operator*=(const Unit & _rhs){
//	//std::string lhsString=toString();
//	//std::string rhsString=_rhs.toString();
//	//std::string parseExpression="("+lhsString+")*("+rhsString+")\n";
//	//(*this)=parseUnit(const_cast<char *> (parseExpression.c_str()));
//
//	kg+=_rhs.kg;
//	m+=_rhs.m;
//	s+=_rhs.s;
//	A+=_rhs.A;
//	K+=_rhs.K;
//	mol+=_rhs.mol;
//	cd+=_rhs.cd;			
//	multiplier*=_rhs.multiplier;
//
//	return *this;
//}
//inline Unit & Unit::operator/=(const Unit & _rhs){
//	//std::string lhsString=toString();
//	//std::string rhsString=_rhs.toString();
//	//std::string parseExpression="("+lhsString+")/("+rhsString+")\n";
//	//(*this)=parseUnit(const_cast<char *> (parseExpression.c_str()));
//
//	kg-=_rhs.kg;
//	m-=_rhs.m;
//	s-=_rhs.s;
//	A-=_rhs.A;
//	K-=_rhs.K;
//	mol-=_rhs.mol;
//	cd-=_rhs.cd;
//	multiplier/=_rhs.multiplier;
//
//	return *this;
//}
//
//inline Unit & Unit::operator*=(double _rhs){
//	//std::stringstream doubleSStream;
//	//doubleSStream<<_rhs;
//	//std::string lhsString=toString();			
//	//std::string parseExpression="("+lhsString+")*"+doubleSStream.str()+"\n";
//	//(*this)=parseUnit(const_cast<char *> (parseExpression.c_str()));
//	multiplier*=_rhs;
//	return *this;
//}
//
//inline Unit & Unit::operator/=(double _rhs){
//	//std::stringstream doubleSStream;
//	//doubleSStream<<_rhs;
//	//std::string lhsString=toString();			
//	//std::string parseExpression="("+lhsString+")/"+doubleSStream.str()+"\n";
//	//(*this)=parseUnit(const_cast<char *> (parseExpression.c_str()));
//	multiplier/=_rhs;
//	return *this;
//}
//inline double Unit::multiplicativeFactor() const{return multiplier;};
//
//
//
//
//
//inline Unit operator*(const Unit & _lhs , const Unit & _rhs) {
//
//	using namespace std;
//	//std::string lhsString=_lhs.toString();
//	//std::string rhsString=_rhs.toString();
//	//std::string parseExpression;
//	//parseExpression="("+lhsString+")*("+rhsString+")\n";
// 	CC3D_Log(LOG_TRACE) << "parseExpression="<<parseExpression;
//	//struct unit_t unit=parseUnit(const_cast<char *> (parseExpression.c_str()));
//
//
//	Unit returnUnit;
//
//	returnUnit.kg=_lhs.kg+_rhs.kg;
//	returnUnit.m=_lhs.m+_rhs.m;
//	returnUnit.s=_lhs.s+_rhs.s;
//	returnUnit.A=_lhs.A+_rhs.A;
//	returnUnit.K=_lhs.K+_rhs.K;
//	returnUnit.mol=_lhs.mol+_rhs.mol;
//	returnUnit.cd=_lhs.cd+_rhs.cd;
//	returnUnit.multiplier=_lhs.multiplier*_rhs.multiplier;
//
//
//	return returnUnit;
//
//}
//
//inline Unit operator/(const Unit & _lhs , const Unit & _rhs) {
//
//	//using namespace std;
//	//std::string lhsString=_lhs.toString();
//	//std::string rhsString=_rhs.toString();
//	//std::string parseExpression;
//	//parseExpression="("+lhsString+")/("+rhsString+")\n";
// 	CC3D_Log(LOG_TRACE) <<"parseExpression="<<parseExpression;
//	//struct unit_t unit=parseUnit(const_cast<char *> (parseExpression.c_str()));
//	//Unit returnUnit(unit);
//	//return returnUnit;
//
//	Unit returnUnit;
//
//	returnUnit.kg=_lhs.kg-_rhs.kg;
//	returnUnit.m=_lhs.m-_rhs.m;
//	returnUnit.s=_lhs.s-_rhs.s;
//	returnUnit.A=_lhs.A-_rhs.A;
//	returnUnit.K=_lhs.K-_rhs.K;
//	returnUnit.mol=_lhs.mol-_rhs.mol;
//	returnUnit.cd=_lhs.cd-_rhs.cd;
//	returnUnit.multiplier=_lhs.multiplier/_rhs.multiplier;
//
//
//	return returnUnit;
//
//
//}
//
//inline Unit operator*(double _lhs , const Unit & _rhs){
//	//std::stringstream doubleSStream;
//	//doubleSStream<<_lhs;
//	//std::string rhsString=_rhs.toString();
//	//std::string parseExpression;
//	//parseExpression=doubleSStream.str()+"*("+rhsString+")\n";
//	//return parseUnit(const_cast<char *> (parseExpression.c_str()));
//
//	Unit returnUnit(_rhs);
//	returnUnit.multiplier*=_lhs;
//	return returnUnit;
//
//
//}
//
//inline Unit operator*(const Unit & _lhs, double _rhs  ){
//	//std::stringstream doubleSStream;
//	//doubleSStream<<_rhs;
//	//std::string lhsString=_lhs.toString();
//	//std::string parseExpression;
//	//parseExpression="("+lhsString+")*"+doubleSStream.str()+"\n";
//	//return parseUnit(const_cast<char *> (parseExpression.c_str()));
//
//	Unit returnUnit(_lhs);
//	returnUnit.multiplier*=_rhs;
//	return returnUnit;
//
//}
//
//inline Unit operator/(double _lhs , const Unit & _rhs){
//	//std::stringstream doubleSStream;
//	//doubleSStream<<_lhs;
//	//std::string rhsString=_rhs.toString();
//	//std::string parseExpression;
//	//parseExpression=doubleSStream.str()+"/("+rhsString+")\n";
//	//return parseUnit(const_cast<char *> (parseExpression.c_str()));
//
//	Unit returnUnit(_rhs);
//	returnUnit.multiplier/=_lhs;
//	return returnUnit;
//
//}
//
//inline Unit operator/(const Unit & _lhs, double _rhs  ){
//	//std::stringstream doubleSStream;
//	//doubleSStream<<_rhs;
//	//std::string lhsString=_lhs.toString();
//	//std::string parseExpression;
//	//parseExpression="("+lhsString+")/"+doubleSStream.str()+"\n";
//	//return parseUnit(const_cast<char *> (parseExpression.c_str()));
//
//	Unit returnUnit(_lhs);
//	returnUnit.multiplier/=_rhs;
//	return returnUnit;
//}
//
//inline Unit powerUnit(const Unit & _lhs,double _power){
//	Unit returnUnit(_lhs);
//	returnUnit.kg*=_power;
//	returnUnit.m*=_power;
//	returnUnit.s*=_power;
//	returnUnit.A*=_power;
//	returnUnit.K*=_power;
//	returnUnit.mol*=_power;
//	returnUnit.cd*=_power;
//	returnUnit.multiplier=pow(returnUnit.multiplier,_power);	
//	return returnUnit;
//}
//
//inline std::ostream & operator<< (std::ostream & _out, const Unit & _unit){
//	using namespace std;
//	_out<<_unit.toString();
//	return _out;
//}

#endif