/*
"kg" | 
 "m" | 
 "s" | 
 "A" | 
 "K" | 
 "mol" | 
 "cd" {currentUnitName=yytext; printf("scanner unit %s\n",currentUnitName) ; return UNIT;} 
*/

/* [0-9]+	{ yylval.d = atoi(yytext); return NUMBER; }*/
/* recognize tokens for the calculator and print them out */
/*[0-9]+"."[0-9]*{EXP}? |
"."?[0-9]+{EXP}? { yylval.d = atof(yytext); printf("this is the number %f\n",atof(yytext)); return NUMBER; }
*/

/*removes the nwecessity to link agains flex library fl*/
%option noyywrap 

%{
#  include <math.h> //have to include this in order for atof or atoi to work. noticve on windows it will compile without this file but will give wrong results
#  include <ctype.h>
# include "unit_calculator.tab.h"
# include <unit_calculator.h>
# include <parserException.h>

char *currentUnitName;
// extern struct unit_t_list_t allocatedUnitList;
extern struct unit_t_list_t * tail;
extern int allocatedUnitNumber=0;



%}

/* float exponent */
EXP	([Ee][-+]?[0-9]+)

%%

"u" |
"d" |
"n" |
"p" |
"f" |
"a" |
"z" |
"y" |
"M" |
"G" |
"T" |
"P" |
"E" |
"Z" |
"Y" |
"k" {yylval.d=power10Symbols(yytext);return NUMBER;} /* dealing with power tem specifiers for units*/


"m" {  /*have to handle the case where m means mili or meter depending on the context */
        int n;
        n=input();
        if(n=="m" || (isalpha(n) && !isdigit(n))){
            yylval.d=power10Symbols(yytext);
            /*printf("got numner %f\n",yylval.d);*/
            unput(n);
            return NUMBER;        
        }
        else{
            yylval.unit=newUnit(yytext); currentUnitName=yytext; 
            /*printf("scanner unit %s\n",currentUnitName) ;*/            
            unput(n);
            return UNIT;
        }
    }

"c" {/*have to handle the case where c means centi or belongs to cd unit name depending on the context */
        int n; 
        n=input();
        if(n=="d"){
            unput(n);
        }
        else{
            yylval.d=power10Symbols(yytext);
            /*printf("got numner %f\n",yylval.d);*/
            unput(n);
            return NUMBER;        
        }
    }

 "g" | 
 "s" | 
 "A" | 
 "K" | 
 "mol" | 
 "cd" {yylval.unit=newUnit(yytext); currentUnitName=yytext; /*printf("scanner unit %s\n",currentUnitName)*/ ; return UNIT;} 

"+"	{ return ADD; }
"-"	{ return SUB; }
"*"	{ return MUL; }
"/"	{ return DIV; }
"^" {return POWER;}
"("     { return OP; }
")"     { return CP; }



[0-9]+"."[0-9]*{EXP}? |
"."?[0-9]+{EXP}? { yylval.d = atof(yytext); /*printf("yytext %s xxx conversion %f\n",yytext,atof(yytext));printf("this is the number %f\n",yylval.d);*/ return NUMBER; }

\n      { return EOL; }
"//".*  
[ \t]   { /* ignore white space */ }
.	{ yyerror("Mystery character %c\n", *yytext);throwParserException("Scanning Error"); }
%%
