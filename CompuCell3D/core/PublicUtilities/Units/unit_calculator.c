#include <unit_calculator.h>
#include <unit_calculator_globals.h>

double power10Symbols(char * _symbol){

    if(!strcmp(_symbol,"u")){
		return 1.E-6;
    }else if(!strcmp(_symbol,"m")){
		return 1.E-3;
    }else if(!strcmp(_symbol,"d")){
		return 1.E-1;
    }else if(!strcmp(_symbol,"c")){
		return 1.E-2;
    }else if(!strcmp(_symbol,"n")){
		return 1.E-9;
    }else if(!strcmp(_symbol,"p")){
		return 1.E-12;
    }else if(!strcmp(_symbol,"f")){
		return 1.E-15;
    }else if(!strcmp(_symbol,"a")){
		return 1.E-18;
    }else if(!strcmp(_symbol,"z")){
		return 1.E-21;
    }else if(!strcmp(_symbol,"y")){
		return 1.E-24;
    }else if(!strcmp(_symbol,"k")){
		return 1.E+3;
    }else if(!strcmp(_symbol,"M")){
		return 1.E+6;
    }else if(!strcmp(_symbol,"G")){
		return 1.E+9;
    }else if(!strcmp(_symbol,"T")){
		return 1.E+12;
    }else if(!strcmp(_symbol,"P")){
		return 1.E+15;
    }else if(!strcmp(_symbol,"E")){
		return 1.E+18;
    }else if(!strcmp(_symbol,"Z")){
		return 1.E+21;
    }else if(!strcmp(_symbol,"Y")){
		return 1.E+24;
    }
    
    return 1.0;
    
}



struct unit_t * newUnit(char * _unitName){
    struct unit_t * unit=malloc(sizeof(struct unit_t));
    addNewAllocatedUnit(&tail,unit);
    //printf("address new unit %d=\n",unit);
    
    unit->kg=unit->m=unit->s=unit->A=unit->K=unit->mol=unit->cd=0.;
    //unit->power10multiplier=0;
    unit->multiplier=1;
	//printf("ARGUMENT UNIT %s\n",_unitName);
    if(!strcmp(_unitName,"g")){
        unit->kg=1.;  
        unit->multiplier=1E-3  ;
        /*printf("got kg unit\n");*/
		return unit;
    }else if (!strcmp(_unitName,"m")){
        unit->m=1.;    
		return unit;        
    }
    else if (!strcmp(_unitName,"s")){
        unit->s=1.;    
		return unit;        
    }    
    else if (!strcmp(_unitName,"A")){
        unit->A=1.;    
		return unit;        
    }    
    else if (!strcmp(_unitName,"K")){
        unit->K=1.;    
		return unit;        
    }    
    else if (!strcmp(_unitName,"mol")){
        unit->mol=1.;    
		return unit;        
    }    
    else if (!strcmp(_unitName,"cd")){
        unit->cd=1.;    
		return unit;        
    }
    return unit;
}

struct unit_t cloneUnit(struct unit_t * _lhs){
    struct unit_t unit;
    unit.kg=_lhs->kg;
    unit.m=_lhs->m;
    unit.s=_lhs->s;
    unit.A=_lhs->A;
    unit.K=_lhs->K;
    unit.mol=_lhs->mol;
    unit.cd=_lhs->cd;
    //unit.power10multiplier=_lhs->power10multiplier;
    unit.multiplier=_lhs->multiplier;
    return unit;
    
}

int multiplyNumbers(int _lhs,int   _rhs){
    return _lhs*_rhs;
}

struct unit_t * multiplyUnits(struct unit_t * _lhs,struct unit_t * _rhs){
    struct unit_t * unit=malloc(sizeof(struct unit_t));
    addNewAllocatedUnit(&tail,unit);
    unit->kg=_lhs->kg+_rhs->kg;
    unit->m=_lhs->m+_rhs->m;
    unit->s=_lhs->s+_rhs->s;
    unit->A=_lhs->A+_rhs->A;
    unit->K=_lhs->K+_rhs->K;
    unit->mol=_lhs->mol+_rhs->mol;
    unit->cd=_lhs->cd+_rhs->cd;
    //unit->power10multiplier=_lhs->power10multiplier+_rhs->power10multiplier;
    unit->multiplier=_lhs->multiplier*_rhs->multiplier;
    return unit;
}

//struct unit_t * multiplyUnitsPower10Multiplier(struct unit_t * _lhs,int _power10multiplier){
//    struct unit_t * unit=malloc(sizeof(struct unit_t));
//    addNewAllocatedUnit(&tail,unit);
//    unit->kg=_lhs->kg;
//    unit->m=_lhs->m;
//    unit->s=_lhs->s;
//    unit->A=_lhs->A;
//    unit->K=_lhs->K;
//    unit->mol=_lhs->mol;
//    unit->cd=_lhs->cd;
//    
//    unit->power10multiplier=_lhs->power10multiplier+(int)log10(_power10multiplier);
//    
//    return unit;
//}

struct unit_t * multiplyUnitsByNumber(struct unit_t * _lhs,double _multiplier){
    struct unit_t * unit=malloc(sizeof(struct unit_t));
    addNewAllocatedUnit(&tail,unit);
    unit->kg=_lhs->kg;
    unit->m=_lhs->m;
    unit->s=_lhs->s;
    unit->A=_lhs->A;
    unit->K=_lhs->K;
    unit->mol=_lhs->mol;
    unit->cd=_lhs->cd;
    
    //unit->power10multiplier=_lhs->power10multiplier+(int)log10(_multiplier);
    //unit->multiplier=_lhs->multiplier*_multiplier/pow(10.0,(int)log10(_multiplier));
	unit->multiplier=_lhs->multiplier*_multiplier;
    return unit;

}


struct unit_t * divideUnits(struct unit_t * _lhs,struct unit_t * _rhs){
    struct unit_t * unit=malloc(sizeof(struct unit_t));
    addNewAllocatedUnit(&tail,unit);
    unit->kg=_lhs->kg-_rhs->kg;
    unit->m=_lhs->m-_rhs->m;
    unit->s=_lhs->s-_rhs->s;
    unit->A=_lhs->A-_rhs->A;
    unit->K=_lhs->K-_rhs->K;
    unit->mol=_lhs->mol-_rhs->mol;
    unit->cd=_lhs->cd-_rhs->cd;
    //unit->power10multiplier=_lhs->power10multiplier-_rhs->power10multiplier;
    //unit->multiplier=_lhs->multiplier/_rhs->multiplier;
	unit->multiplier=_lhs->multiplier/_rhs->multiplier;
    return unit;
}

struct unit_t * divideNumberByUnit(double _multiplier,struct unit_t * _rhs){
    struct unit_t * unit=malloc(sizeof(struct unit_t));
    addNewAllocatedUnit(&tail,unit);
    unit->kg= - _rhs->kg;
    unit->m= - _rhs->m;
    unit->s= - _rhs->s;
    unit->A= - _rhs->A;
    unit->K= - _rhs->K;
    unit->mol= - _rhs->mol;
    unit->cd= - _rhs->cd;
    
    // unit->power10multiplier=_lhs->power10multiplier+(int)log10(_multiplier);
    // unit->multiplier=_lhs->multiplier*_multiplier/pow(10.0,(int)log10(_multiplier));

    
    //unit->power10multiplier=(int)log10(_multiplier) - _rhs->power10multiplier;
    //unit->multiplier=(_multiplier/pow(10.0,(int)log10(_multiplier)))/_rhs->multiplier;

	unit->multiplier=_multiplier/_rhs->multiplier;
    return unit;
}

struct unit_t * powUnit(struct unit_t * _lhs,double _pow){
    struct unit_t * unit=malloc(sizeof(struct unit_t));
    addNewAllocatedUnit(&tail,unit);
    unit->kg=_lhs->kg*_pow;
    unit->m=_lhs->m*_pow;
    unit->s=_lhs->s*_pow;
    unit->A=_lhs->A*_pow;
    unit->K=_lhs->K*_pow;
    unit->mol=_lhs->mol*_pow;
    unit->cd=_lhs->cd*_pow;

 //   unit->power10multiplier=_lhs->power10multiplier*_pow;
	//unit->multiplier=_lhs->multiplier;

    unit->multiplier=pow(_lhs->multiplier,_pow);
    return unit;


}


 void printUnit(struct unit_t * _lhs){
    int power10Multiplier=(int)log10(_lhs->multiplier);
	double multiplier=_lhs->multiplier/pow(10,power10Multiplier);

    printf("%.4g*10^%d * kg^%.4g * m^%.4g * s^%.4g * A^%.4g * K^%.4g * mol^%.4g * cd^%.4g\n",multiplier,power10Multiplier,_lhs->kg,_lhs->m,_lhs->s,_lhs->A,_lhs->K,_lhs->mol,_lhs->cd);
    // printf ("%f*10^%d\n",_lhs->multiplier,_lhs->power10multiplier);
 }

 
 void addNewAllocatedUnit(struct unit_t_list_t ** _listTail, struct unit_t *_unitPtr){  
    struct unit_t_list_t * next_list_elem=malloc(sizeof(struct unit_t_list_t));
    
    next_list_elem->previous=*_listTail;
    *_listTail=next_list_elem;
    (*_listTail)->next=0;
    
    //_unitPtr->allocatedUnitNumber=allocatedUnitNumber++;
    // printf("ADDING unit number=%d address %d\n",_unitPtr->allocatedUnitNumber,_unitPtr);
    (*_listTail)->unitPtr=_unitPtr;    
    //tail=_listTail;
    // printf("add tail %d _listTail=%d\n",tail,*_listTail);
    

}

void freeList(struct unit_t_list_t * _list){
    struct unit_t_list_t *removedElement;
    struct unit_t_list_t *currElemPtr=_list;    
    // printf("address current unit %d\n",currElemPtr->unitPtr);
    while(currElemPtr->previous){
        
        
        if(currElemPtr->unitPtr){
            // printf("FREE MEMORY %d\n",currElemPtr->unitPtr);        
            free(currElemPtr->unitPtr);
            currElemPtr->unitPtr=0;
            removedElement=currElemPtr;
            currElemPtr=currElemPtr->previous;
            // printf("freeing %d element\n",removedElement);
            free(removedElement);
            
        }
    }
        if(currElemPtr->unitPtr){
            // printf("FREE MEMORY %d\n",currElemPtr->unitPtr);        
            free(currElemPtr->unitPtr);
            currElemPtr->unitPtr=0;    
            removedElement=currElemPtr;
            currElemPtr=currElemPtr->previous;
            // printf("freeing %d element\n",removedElement);
            free(removedElement);
        }
    


    
    // while(currElemPtr->next){
        // printf("FREE MEMORY %d\n",currElemPtr->next->unitPtr);        
        // currElemPtr=currElemPtr->next;
        // // if(currElemPtr->unitPtr){
            // // printf("freeing unit number %d address %d\n",currElemPtr->unitPtr->allocatedUnitNumber,currElemPtr->unitPtr);
            // // free(currElemPtr->unitPtr);
            // // currElemPtr->unitPtr=0;
            // // removedElement=currElemPtr;
            // // currElemPtr=currElemPtr->next;
            // // free(removedElement);
        // // }
    // }
    // while(currElemPtr->previous){
        // printf("FREE MEMORY %d\n",currElemPtr->unitPtr);
        // if(currElemPtr->unitPtr){
            // printf("freeing unit number %d address %d\n",currElemPtr->unitPtr->allocatedUnitNumber,currElemPtr->unitPtr);
            // free(currElemPtr->unitPtr);
            // currElemPtr->unitPtr=0;
            // removedElement=currElemPtr;
            // currElemPtr=currElemPtr->previous;
            // free(removedElement);
        // }
    // }    
}


 // struct unit_t newUnit(char * _unitName){
    // struct unit_t unit;
    // unit.kg=unit.m=unit.s=unit.A=unit.K=unit.mol=unit.cd=0;
	// printf("ARGUMENT UNIT %s\n",_unitName);
    // if(!strcmp(_unitName,"kg")){
        // unit.kg=1;  
        // printf("got kg unit\n");
		// return unit;
    // }else if (!strcmp(_unitName,"m")){
        // unit.m=1;    
		// return unit;        
    // }
    // else if (!strcmp(_unitName,"s")){
        // unit.s=1;    
		// return unit;        
    // }    
    // else if (!strcmp(_unitName,"A")){
        // unit.A=1;    
		// return unit;        
    // }    
    // else if (!strcmp(_unitName,"K")){
        // unit.K=1;    
		// return unit;        
    // }    
    // else if (!strcmp(_unitName,"mol")){
        // unit.mol=1;    
		// return unit;        
    // }    
    // else if (!strcmp(_unitName,"cd")){
        // unit.cd=1;    
		// return unit;        
    // }
    // return unit;
// }

 
// struct unit_t multiplyUnits(struct unit_t _lhs,struct unit_t _rhs){
    // unit_t unit;
    // unit.kg=_lhs.kg+_rhs.kg;
    // unit.m=_lhs.m+_rhs.m;
    // unit.s=_lhs.s+_rhs.s;
    // unit.A=_lhs.A+_rhs.A;
    // unit.K=_lhs.K+_rhs.K;
    // unit.mol=_lhs.mol+_rhs.mol;
    // unit.cd=_lhs.cd+_rhs.cd;
    // return unit;
// }
// struct unit_t divideUnits(struct unit_t _lhs,struct unit_t _rhs){
    // unit_t unit;
    // unit.kg=_lhs.kg-_rhs.kg;
    // unit.m=_lhs.m-_rhs.m;
    // unit.s=_lhs.s-_rhs.s;
    // unit.A=_lhs.A-_rhs.A;
    // unit.K=_lhs.K-_rhs.K;
    // unit.mol=_lhs.mol-_rhs.mol;
    // unit.cd=_lhs.cd-_rhs.cd;
    // return unit;
// }
