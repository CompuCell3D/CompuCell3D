#ifndef unit_calculator_h
#define unit_calculator_h

#ifdef __cplusplus
extern "C" {
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

struct unit_t {
  double kg;
  double m;
  double s;
  double A;
  double K;
  double mol;
  double cd;
  //int power10multiplier;
  double multiplier;
  //int allocatedUnitNumber;
};

typedef struct unit_t Unit_t;

struct unit_t_list_t {
    struct unit_t *unitPtr;
    struct unit_t_list_t *next;    
    struct unit_t_list_t *previous;
};


extern struct unit_t_list_t allocatedUnitList;
extern struct unit_t_list_t * tail;
extern int allocatedUnitNumber;
//allocatedUnitList.ptr=allocatedUnitList.next=0;

void addNewAllocatedUnit(struct unit_t_list_t ** _listTail, struct unit_t *_unitPtr);

void freeList(struct unit_t_list_t * _list);
double power10Symbols(char * _symbol);


// struct unit_t newUnit(char * _unitName);

 // struct unit_t  multiplyUnits(struct unit_t  _lhs,struct unit_t  _rhs);
 // struct unit_t  divideUnits(struct unit_t _lhs,struct unit_t _rhs);

 struct unit_t * newUnit(char * _unitName);
 
 struct unit_t * multiplyUnits(struct unit_t * _lhs,struct unit_t  * _rhs);
 struct unit_t * divideUnits(struct unit_t * _lhs,struct unit_t * _rhs);
 //struct unit_t * multiplyUnitsPower10Multiplier(struct unit_t * _lhs,int _power10multiplier);
 struct unit_t * multiplyUnitsByNumber(struct unit_t * _lhs,double _multiplier);
 struct unit_t * divideNumberByUnit(double _multiplier,struct unit_t * _rhs);
 struct unit_t * powUnit(struct unit_t * _lhs,double pow);
 
 struct unit_t cloneUnit(struct unit_t * _lhs);
 
 void printUnit(struct unit_t * _lhs);

 int multiplyNumbers(int _lhs,int  _rhs);
 
 struct unit_t parseUnit(char * _unitStr);
 
 #ifdef __cplusplus
}
#endif

#endif 