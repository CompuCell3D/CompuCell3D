
/* simplest version of calculator */

%{
#  include <stdio.h>
#  include <math.h> //have to include this in order for atof or atoi to work. noticve on windows it will compile without this file but will give wrong results
# include <unit_calculator.h>
# include <unit_calculator_main_lib.h>
struct unit_t  returnedUnit;

//typedef struct unit_t UnitType;
%}

%union {
  struct unit_t * unit;
  double d;
}

/* declare tokens */
%token UNIT
%token NUMBER
%token ADD SUB MUL DIV ABS
%token POWER
%token OP CP
%token EOL

%type <d> factor exp
%type <d> term powerterm

%type <unit> unitfactor
%type <unit> unitterm 
%type <unit> unitpowerterm





%%

calclist: /* nothing */
 | calclist exp EOL { printf("= %.4g\n> ", $2); }
 | calclist unitfactor EOL {                               
                            { //have to put pure C code in the {} otherwise cannot define variables e.g. 'double a';
                               // printUnit($2);
                               struct unit_t  checkUnit=cloneUnit($2);
                               freeList(tail);
                               tail=0;     
                               // printf("\n>");
                               returnedUnit=checkUnit;
                             }
                            }
 | calclist EOL { printf("> "); } /* blank line or a comment */
 ;

exp: factor
 | exp ADD exp { $$ = $1 + $3; }
 | exp SUB factor { $$ = $1 - $3; } 
 ; 
 
factor: powerterm
 | factor MUL powerterm { $$ = $1 * $3; } 
 | factor DIV powerterm { $$ = $1 / $3; }
 ;
 


powerterm: term
| term POWER powerterm  {$$=pow($1,$3);} //right associativity - this is standard
 // | powerterm POWER term  {$$=(int)pow($1,$3);} //left associativity
; 
 
term: NUMBER 
 | OP exp CP { $$ = $2; }
 | SUB term    { $$ = -$2 }
 ;
 
 
 
 unitfactor: unitpowerterm  
  | unitfactor MUL unitpowerterm { $$ = multiplyUnits($1 , $3); }
  | unitfactor DIV unitpowerterm { $$ = divideUnits($1 , $3); }
  | unitfactor MUL powerterm { /*printf("multiplying unig  and %f\n",$3);*/$$ = multiplyUnitsByNumber($1 , $3); } 
  | factor MUL unitpowerterm { $$ = multiplyUnitsByNumber($3 , $1); }  
  | unitfactor DIV powerterm { $$ = multiplyUnitsByNumber($1 , 1.0/$3); } 
  // | factor DIV unitpowerterm { $$ = multiplyUnitsByNumber(powUnit($3,-1) , $1); }  
   | factor DIV unitpowerterm { $$ = divideNumberByUnit($1,$3); }  
  ;

  
unitpowerterm: 
 //  term
 unitterm
 // | unitterm POWER term  {printf("unitterm POWER term\n");$$=powUnit($1,$3);}
  | unitpowerterm POWER term {/*printf("unitpowerterm POWER term\n");*/$$=powUnit($1,$3);}
  | term unitterm {$$=multiplyUnitsByNumber($2 , $1);}

; 
  
 unitterm: UNIT 
 //{printf ("parser unit %d\n",$1); /*$$ = newUnit($1);*/}
  | OP unitfactor CP { $$ = $2; }  
 ;

 
 
%%


struct unit_t parseUnit(char * _unitStr){
  struct unit_t ;
  tail=0;
  // printf("this is flowting point number %g\n",atof("10"));
  // exit(1);
  printf("> "); 
  //char ptr; 
  yy_scan_string(_unitStr); //have to include \n ot \0 at the end of the string to signal to scanner EOF
  // yy_scan_string("(2+2)*3*2*kg*kg^(2/3)*(2+6)*s*0.5*10^2*mol^2*10^2\n"); //have to include \n at the end of the string to signal to scanner EOF
  // char unitPtr;
  yyparse();
  return returnedUnit;
  
}



// main()
// {
    
    // char * unitString="(2+2)*3*2*kg*kg^(2/3)*(2+6)*s*0.5*10^2*mol^2*10^2\n";
    // parseUnit(unitString);
    // printf("returnedUnit\n");
    // printUnit(&returnedUnit);
    



  // // // char * unitString="(2+2)*3*2*kg*kg^(2/3)*(2+6)*s*0.5*10^2*mol^2*10^2\n";
  // // // parseUnit(unitString);
  // // tail=0;
  // // // printf("this is flowting point number %g\n",atof("10"));
  // // // exit(1);
  
  // // printf("> "); 
  // // // yy_scan_string("(2+2)*3*2*kg*kg^(2/3)*(2+6)*s*0.5*10^2*mol^2*10^2\n"); //have to include \n at the end of the string to signal to scanner EOF
  // // yyparse();
  
// }

yyerror(char *s)
{  
  fprintf(stderr, "error: %s\n", s);
  throwParserException("SYNTAX ERROR");
}
