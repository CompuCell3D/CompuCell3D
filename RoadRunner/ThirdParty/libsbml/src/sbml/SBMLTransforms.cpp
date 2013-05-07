/**
 * @file    SBMLTransforms.cpp
 * @brief   Transform functions
 * @author  Sarah Keating
 * 
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 *
 * Copyright (C) 2009-2012 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EBML-EBI), Hinxton, UK
 *  
 * Copyright (C) 2006-2008 by the California Institute of Technology,
 *     Pasadena, CA, USA 
 *  
 * Copyright (C) 2002-2005 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. Japan Science and Technology Agency, Japan
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <sbml/SBMLTransforms.h>
#include <sbml/Model.h>

#include <cstring>
#include <cmath>

#if defined(_MSC_VER) || defined(__BORLANDC__)
#include <float.h>
#  define isnan(d)  _isnan(d)
#endif

#if defined(__linux)
#define isnan std::isnan
#endif
#include "validator/constraints/IdList.h"

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

/** @cond doxygen-libsbml-internal */
SBMLTransforms::IdValueMap SBMLTransforms::mValues;

void
SBMLTransforms::replaceFD(ASTNode * node, const ListOfFunctionDefinitions *lofd)
{
  bool replaced = false;

  /* write a list of fd ids */
  IdList ids;
  unsigned int i;
  for (i = 0; i < lofd->size(); i++)
  {
    ids.append(lofd->get(i)->getId());
  }
  
  /* if any of these ids exist in the ASTnode replace */
  /* Need a get out if replace fails - shouldnt happen but as a fail-safe */ 
  unsigned int count = 0;
  do 
  {
    for (i = 0; i < lofd->size(); i++)
    {
      replaceFD(node, lofd->get(i));
    }

    replaced = !(checkFunctionNodeForIds(node, ids));
    count++;
  } 
  while (!replaced && count < 2*lofd->size());
}

void
SBMLTransforms::replaceFD(ASTNode * node, const FunctionDefinition *fd)
{
  if ((node == NULL) || (fd == NULL))
    return;
  
  if (node->isFunction() && node->getName() == fd->getId())
  {
   replaceBvars(node, fd);
   for (unsigned int j = 0; j < node->getNumChildren(); j++)
   {
     replaceFD(node->getChild(j), fd);
   }
  }
  else
  {
    for (unsigned int i = 0; i < node->getNumChildren(); i++)
    {
      replaceFD(node->getChild(i), fd);
    }
  }
}
void
SBMLTransforms::replaceBvars(ASTNode * node, const FunctionDefinition *fd)
{
  ASTNode * fdMath = NULL;
  unsigned int noBvars;

  if (fd != NULL && fd->isSetMath())
    {
      noBvars = fd->getNumArguments();
      if (noBvars == 0)
      {
        fdMath = fd->getMath()->getLeftChild()->deepCopy();
      }
      else
      {
        fdMath = fd->getMath()->getRightChild()->deepCopy();
      }
   
      for (unsigned int i = 0, nodeCount = 0; i < noBvars; i++, nodeCount++)
      {
        if (nodeCount < node->getNumChildren())
          fdMath->replaceArgument(fd->getArgument(i)->getName(), 
                                            node->getChild(nodeCount));
      }
    }
    (*node) = *fdMath;
}


bool
SBMLTransforms::checkFunctionNodeForIds(ASTNode * node, IdList& ids)
{
  bool present = false;
  unsigned int i = 0;
  unsigned int numChildren = node->getNumChildren();

  if (node != NULL && node->getType() == AST_FUNCTION)
  {
    if (ids.contains(node->getName()))
    {
      present = true;
    }
  }

  while (!present && i < numChildren)
  {
    present = checkFunctionNodeForIds(node->getChild(i), ids);
    i++;
  }
  
  return present;
}


bool
SBMLTransforms::nodeContainsId(const ASTNode * node, IdList& ids)
{
  bool present = false;
  unsigned int i = 0;
  unsigned int numChildren = node->getNumChildren();

  if (node != NULL && node->getType() == AST_NAME)
  {
    if (ids.contains(node->getName()))
    {
      present = true;
    }
  }

  while (!present && i < numChildren)
  {
    present = nodeContainsId(node->getChild(i), ids);
    i++;
  }
  
  return present;
}

bool
SBMLTransforms::nodeContainsNameNotInList(const ASTNode * node, IdList& ids)
{
  bool notInList = false;
  unsigned int i = 0;
  unsigned int numChildren = node->getNumChildren();

  if (node != NULL && node->getType() == AST_NAME)
  {
    if (!(ids.contains(node->getName())))
    {
      notInList = true;
    }
  }

  while (!notInList && i < numChildren)
  {
    notInList = nodeContainsNameNotInList(node->getChild(i), ids);
    i++;
  }
  
  return notInList;
}

IdList 
SBMLTransforms::mapComponentValues(const Model * m)
{
  mValues.clear();
  /* it is possible that a model does not have all 
   * the necessary values specified
   * in which case we can not calculate other values
   * keep a list so we can check
   */
  IdList ids;

  unsigned int i, j;
  for (i = 0; i < m->getNumCompartments(); i++)
  {
    const Compartment* c = m->getCompartment(i);

    /* is value assigned by an assignmentRule
     * or an initialAssignment
     * or specified
     * - if none then the model is incomplete
     */
    const Rule * r = m->getRule(c->getId());
    if (((r == NULL) || (r->getType() == RULE_TYPE_RATE))
      && (m->getInitialAssignment(c->getId()) == NULL))
    {
      /* not set by assignment */
      if (!(c->isSetSize()))
      {
        ids.append(c->getId());
        ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), false);
        mValues.insert(pair<const std::string, ValueSet>(c->getId(), v));
      }
      else
      {
        ValueSet v = make_pair(c->getSize(), true);
        mValues.insert(pair<const std::string, ValueSet>(c->getId(), v));
      }
    }
    else
    {
      /* is set by assignment - need to work it out */
      ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), true);
      mValues.insert(pair<const std::string, ValueSet>(c->getId(), v));
    }
  }

  for (i = 0; i < m->getNumSpecies(); i++)
  {
    const Species * s = m->getSpecies(i);
    /* is value assigned by an assignmentRule
     * or an initialAssignment
     * or specified
     * - if none then the model is incomplete
     */
    const Rule * r = m->getRule(s->getId());
    if (((r == NULL) || (r->getType() == RULE_TYPE_RATE))
      && (m->getInitialAssignment(s->getId()) == NULL))
    {
      if (!(s->isSetInitialAmount()) && !(s->isSetInitialConcentration()))
      {
        ids.append(s->getId());
        ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), false);
        mValues.insert(pair<const std::string, ValueSet>(s->getId(), v));
      }
      else
      {
        /* here need to be careful as the id will refer to concentration
         * unless hasOnlySubstanceUnits is true
         * regardless of whether amount/concentration has been set
         */
        if (s->getHasOnlySubstanceUnits())
        {
          ValueSet v = make_pair(s->getInitialAmount(), true);
          mValues.insert(pair<const std::string, ValueSet>(s->getId(), v));
          //mValues.insert(pair<const std::string, double>(s->getId(), 
          //                                            s->getInitialAmount()));
        }
        else if (s->isSetInitialAmount())
        {
          /* at present only deal with case where compartment size is fixed */
          IdValueIter it;
          it = mValues.find(s->getCompartment());
          if (it != mValues.end())
          {
            /* compartment size is set */
            if (((*it).second).second)
            {
              double conc = s->getInitialAmount()/(((*it).second).first);
              ValueSet v = make_pair(conc, true);
              mValues.insert(pair<const std::string, ValueSet>(s->getId(), v));
            }
            else
            {
              ids.append(s->getId());
              ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), false);
              mValues.insert(pair<const std::string, ValueSet>(s->getId(), v));
            }
          }
          else
          {
            ids.append(s->getId());
            ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), false);
            mValues.insert(pair<const std::string, ValueSet>(s->getId(), v));
          }
        }
        else
        {
          ValueSet v = make_pair(s->getInitialConcentration(), true);
          mValues.insert(pair<const std::string, ValueSet>(s->getId(), v));
        }

      }
    }
    else
    {
      /* is set by assignment - need to work it out */
      ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), true);
      mValues.insert(pair<const std::string, ValueSet>(s->getId(), v));
    }
  }

  for (i = 0; i < m->getNumParameters(); i++)
  {
    const Parameter * p = m->getParameter(i);

    /* is value assigned by an assignmentRule
     * or an initialAssignment
     * or specified
     * - if none then the model is incomplete
     */
    const Rule * r = m->getRule(p->getId());
    if (((r == NULL) || (r->getType() == RULE_TYPE_RATE))
      && (m->getInitialAssignment(p->getId()) == NULL))
    {
      if (!(p->isSetValue()))
      {
        ids.append(p->getId());
        ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), false);
        mValues.insert(pair<const std::string, ValueSet>(p->getId(), v));
      }
      else
      {
        ValueSet v = make_pair(p->getValue(), true);
        mValues.insert(pair<const std::string, ValueSet>(p->getId(), v));
      }
    }
    else
    {
      /* is set by assignment - need to work it out */
      ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), true);
      mValues.insert(pair<const std::string, ValueSet>(p->getId(), v));
    }
  }

  for (i = 0; i < m->getNumReactions(); i++)
  {
    const Reaction *rn = m->getReaction(i);

    for (j = 0; j < rn->getNumReactants(); j++)
    {
      const SpeciesReference * sr = rn->getReactant(j);

      /* is value assigned by an assignmentRule
      * or an initialAssignment
      * or specified
      * - if none then the model is incomplete
      */
      const Rule * r = m->getRule(sr->getId());
      if (((r == NULL) || (r->getType() == RULE_TYPE_RATE))
        && (m->getInitialAssignment(sr->getId()) == NULL))
      {
        /* not set by assignment */
        if (!(sr->isSetStoichiometry()))
        {
          ids.append(sr->getId());
          ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), false);
          mValues.insert(pair<const std::string, ValueSet>(sr->getId(), v));
        }
        else
        {
          ValueSet v = make_pair(sr->getStoichiometry(), true);
          mValues.insert(pair<const std::string, ValueSet>(sr->getId(), v));
        }
      }
      else
      {
        /* is set by assignment - need to work it out */
        ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), true);
        mValues.insert(pair<const std::string, ValueSet>(sr->getId(), v));
      }
    }

    for(j = 0; j < rn->getNumProducts(); j++)
    {
      const SpeciesReference * sr = rn->getProduct(j);
      /* is value assigned by an assignmentRule
      * or an initialAssignment
      * or specified
      * - if none then the model is incomplete
      */
      const Rule * r = m->getRule(sr->getId());
      if (((r == NULL) || (r->getType() == RULE_TYPE_RATE))
        && (m->getInitialAssignment(sr->getId()) == NULL))
      {
        /* not set by assignment */
        if (!(sr->isSetStoichiometry()))
        {
          ids.append(sr->getId());
          ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), false);
          mValues.insert(pair<const std::string, ValueSet>(sr->getId(), v));
        }
        else
        {
          ValueSet v = make_pair(sr->getStoichiometry(), true);
          mValues.insert(pair<const std::string, ValueSet>(sr->getId(), v));
        }
      }
      else
      {
        /* is set by assignment - need to work it out */
        ValueSet v = make_pair(numeric_limits<double>::quiet_NaN(), true);
        mValues.insert(pair<const std::string, ValueSet>(sr->getId(), v));
      }
    }
  }

  /* returns a list of ids for which the model has no declared value
   * and no assignment
   */
  return ids;
}


void 
SBMLTransforms::clearComponentValues()
{
  mValues.clear();
}



double
SBMLTransforms::evaluateASTNode(const ASTNode *node, const Model *m)
{

  double result;
  int i;

  switch (node->getType())
  {
  case AST_INTEGER:
    result = (double) (node->getInteger());
    break;

  case AST_REAL:
    result = node->getReal();
    break;

  case AST_REAL_E:
    result = node->getReal();
    break;

  case AST_RATIONAL:
    result = node->getReal();
    break;
  
  case AST_NAME:
    if (!mValues.empty())
    {
      if (mValues.find(node->getName()) != mValues.end())
      {
        result = (mValues.find(node->getName())->second).first;
        bool set = (mValues.find(node->getName())->second).second;
        if (isnan(result) && set)
        {
          if (m != NULL)
          {
            // means the value is set by a rule/initialAssignment
            const Rule *r = m->getRule(node->getName());
            const InitialAssignment *ia = 
                                m->getInitialAssignment(node->getName());
            if (r != NULL)
            {
              result = evaluateASTNode(r->getMath(), m);
            }
            else if (ia != NULL)
            {
              result = evaluateASTNode(ia->getMath(), m);
            }
          }
        }
      }
      else
      {
        result = numeric_limits<double>::quiet_NaN();
      }
    }
    else
      result = numeric_limits<double>::quiet_NaN();
    break;

  case AST_NAME_AVOGADRO:
    result = node->getReal();
    break;

  case AST_NAME_TIME:
    result = 0.0;
    break;

  case AST_CONSTANT_E:
    /* exp(1) is used to adjust exponentiale to machine precision */
    result = exp(1.0);
    break;

  case AST_CONSTANT_FALSE:
    result = 0.0;
    break;

  case AST_CONSTANT_PI:
    /* pi = 4 * atan 1  is used to adjust Pi to machine precision */
    result = 4.0*atan(1.0);
    break;

  case AST_CONSTANT_TRUE:
    result = 1.0;
    break;

  case AST_LAMBDA:
  case AST_FUNCTION:
    /* shouldnt get here */
    result = numeric_limits<double>::quiet_NaN();
    break;

  case AST_PLUS:
    if (node->getNumChildren() == 0)
    {
      result = 0.0;
    }
    else if (node->getNumChildren() == 1)
    {
      result = evaluateASTNode(node->getChild(0), m);
    }
    else
    {
      result = evaluateASTNode(node->getChild(0), m) + 
               evaluateASTNode(node->getChild(1), m) ;
    }
    break;

  case AST_MINUS:
    if(node->getNumChildren() == 1)
      result = - (evaluateASTNode(node->getChild(0), m));
    else
    result = evaluateASTNode(node->getChild(0), m) - 
             evaluateASTNode(node->getChild(1), m) ;
    break;

  case AST_TIMES:
    if (node->getNumChildren() == 0)
    {
      result = 1.0;
    }
    else if (node->getNumChildren() == 1)
    {
      result = evaluateASTNode(node->getChild(0), m);
    }
    else
    {
      result = evaluateASTNode(node->getChild(0), m) * 
               evaluateASTNode(node->getChild(1), m) ;
    }
    break;

  case AST_DIVIDE:
    result = evaluateASTNode(node->getChild(0), m) / 
             evaluateASTNode(node->getChild(1), m) ;
    break;

  case AST_POWER:
  case AST_FUNCTION_POWER:
    result = pow(evaluateASTNode(node->getChild(0), m), 
             evaluateASTNode(node->getChild(1), m)) ;
    break;

  case AST_FUNCTION_ABS:
    result = (double) (fabs((double)(evaluateASTNode(node->getChild(0), m))));
    break;

  case AST_FUNCTION_ARCCOS:
    result = acos(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_ARCCOSH:
    /* arccosh(x) = ln(x + sqrt(x-1).sqrt(x+1)) */
    result = log(evaluateASTNode(node->getChild(0), m)
      + pow((evaluateASTNode(node->getChild(0), m)-1), 0.5)
      * pow((evaluateASTNode(node->getChild(0), m)+1), 0.5));
    break;

  case AST_FUNCTION_ARCCOT:
    /* arccot x =  arctan (1 / x) */
    result = atan(1.0/evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_ARCCOTH:
    /* arccoth x = 1/2 * ln((x+1)/(x-1)) */
    result = ((1.0/2.0) * log((evaluateASTNode(node->getChild(0), m) + 1.0)
                             /(evaluateASTNode(node->getChild(0), m) - 1.0)) );
    break;

  case AST_FUNCTION_ARCCSC:
    /* arccsc(x) = Arcsin(1 / x) */
    result = asin( 1.0/evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_ARCCSCH:
    /* arccsch(x) = ln((1 + sqrt(1 + x^2)) / x) */
    result = log((1.0 + pow(1.0 + 
      pow(evaluateASTNode(node->getChild(0), m), 2), 0.5))
      /evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_ARCSEC:
    /* arcsec(x) = arccos(1/x) */
    result = acos(1.0/evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_ARCSECH:
    /* arcsech(x) = ln((1 + sqrt(1 - x^2)) / x) */
    result = log((1.0 + pow((1.0 - 
      pow(evaluateASTNode(node->getChild(0), m), 2)), 0.5))
      /evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_ARCSIN:
    result = asin(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_ARCSINH:
    /* arcsinh(x) = ln(x + sqrt(1 + x^2)) */
    result = log(evaluateASTNode(node->getChild(0), m)
      + pow((1.0+pow(evaluateASTNode(node->getChild(0), m), 2)), 0.5));
    break;

  case AST_FUNCTION_ARCTAN:
    result = atan(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_ARCTANH:
    /* arctanh = 0.5 * ln((1+x)/(1-x)) */
    result = 0.5 * log((1.0 + evaluateASTNode(node->getChild(0), m))
      /(1.0 - evaluateASTNode(node->getChild(0), m)));
    break;

  case AST_FUNCTION_CEILING:
    result = ceil(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_COS:
    result = cos(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_COSH:
    result = cosh(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_COT:
    /* cot x = 1 / tan x */
    result = (1.0/tan(evaluateASTNode(node->getChild(0), m)));
    break;

  case AST_FUNCTION_COTH:
    /* coth x = cosh x / sinh x */
    result = cosh(evaluateASTNode(node->getChild(0), m)) / 
             sinh(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_CSC:
    /* csc x = 1 / sin x */
    result = (1.0/sin(evaluateASTNode(node->getChild(0), m)));
    break;

  case AST_FUNCTION_CSCH:
    /* csch x = 1 / sinh x  */
    result = (1.0/sinh(evaluateASTNode(node->getChild(0), m)));
    break;

  case AST_FUNCTION_DELAY:
    result = numeric_limits<double>::quiet_NaN();
    break;

  case AST_FUNCTION_EXP:
    result = exp(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_FACTORIAL:
    i = (int) (floor(evaluateASTNode(node->getChild(0), m)));
    for(result=1; i>1; --i)
    {
      result *= i;
    }
    break;

  case AST_FUNCTION_FLOOR:
    result = floor(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_LN:
    result = log(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_LOG:
    result = log10(evaluateASTNode(node->getChild(1), m));
    break;

  case AST_FUNCTION_PIECEWISE:    
    {
      unsigned int numChildren = node->getNumChildren();
      if ( numChildren % 2 == 0)
      {
        bool assigned = false;
        for (unsigned int i = 0; i < numChildren; i+=2)
        {
          // compute piece
          double value = evaluateASTNode(node->getChild(i), m);
          double boolean = evaluateASTNode(node->getChild(i+1), m);
          if (boolean == 1.0)
          {
            // we might have two true piece statements
            // if the values are the same - fine
            // if not then the result is undefined
            if (assigned == true)
            {
              if (value == result)
              {
                continue;
              }
              else
              {
                result = numeric_limits<double>::quiet_NaN();
              }
            }
            else
            {
              result = value;
              assigned = true;
            }
          }
        }
        if (!assigned)
        {
          result = numeric_limits<double>::quiet_NaN();
        }
      }
      else
      {
        bool assigned = false;
        for (unsigned int i = 0; i < numChildren-1; i+=2)
        {
          // compute piece
          double value = evaluateASTNode(node->getChild(i), m);
          double boolean = evaluateASTNode(node->getChild(i+1), m);
          if (boolean == 1.0)
          {
            // we might have two true piece statements
            // if the values are the same - fine
            // if not then the result is undefined
            if (assigned == true)
            {
              if (value == result)
              {
                continue;
              }
              else
              {
                result = numeric_limits<double>::quiet_NaN();
              }
            }
            else
            {
              result = value;
              assigned = true;
            }
           }
        }
        if (!assigned)
        {
          // compute otherwise
          result = evaluateASTNode(node->getChild(numChildren-1), m);
        }
      }
    }
    break;

  case AST_FUNCTION_ROOT:
    result = pow(evaluateASTNode(node->getChild(1), m),
      (1.0/evaluateASTNode(node->getChild(0), m)));
    break;

  case AST_FUNCTION_SEC:
    /* sec x = 1 / cos x */
    result = 1.0/cos(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_SECH:
    /* sech x = 1 / cosh x */
    result = 1.0/cosh(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_SIN:
    result = sin(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_SINH:
    result = sinh(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_TAN:
    result = tan(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_FUNCTION_TANH:
    result = tanh(evaluateASTNode(node->getChild(0), m));
    break;

  case AST_LOGICAL_AND:
    {
      if (node->getNumChildren() == 0)
        result = 1.0;
      else if (node->getNumChildren() == 1)
        result = evaluateASTNode(node->getChild(0), m);
      else
        result = (double) ((evaluateASTNode(node->getChild(0), m)) 
        && (evaluateASTNode(node->getChild(1), m)));
    }
    break;

  case AST_LOGICAL_NOT:
    result = (double) (!(evaluateASTNode(node->getChild(0), m)));
    break;

  case AST_LOGICAL_OR:
    {
      if (node->getNumChildren() == 0)
        result = 0.0;
      else if (node->getNumChildren() == 1)
        result = evaluateASTNode(node->getChild(0), m);
      else
        result = (double) ((evaluateASTNode(node->getChild(0), m)) 
        || (evaluateASTNode(node->getChild(1), m)));
    }
    break;

  case AST_LOGICAL_XOR:
    {
      if (node->getNumChildren() == 0)
        result = 0.0;
      else if (node->getNumChildren() == 1)
        result = evaluateASTNode(node->getChild(0), m);
      else
        result = (double) ((!(evaluateASTNode(node->getChild(0), m)) 
                       && (evaluateASTNode(node->getChild(1), m)))
        || ((evaluateASTNode(node->getChild(0), m)) 
                       && !(evaluateASTNode(node->getChild(1), m))));
    }
    break;

  case AST_RELATIONAL_EQ :
    result = (double) ((evaluateASTNode(node->getChild(0), m)) 
      == (evaluateASTNode(node->getChild(1), m)));
    break;

  case AST_RELATIONAL_GEQ:
    result = (double) ((evaluateASTNode(node->getChild(0), m)) 
      >= (evaluateASTNode(node->getChild(1), m)));
    break;

  case AST_RELATIONAL_GT:
    result = (double) ((evaluateASTNode(node->getChild(0), m)) 
      > (evaluateASTNode(node->getChild(1), m)));
    break;

  case AST_RELATIONAL_LEQ:
    result = (double) ((evaluateASTNode(node->getChild(0), m)) 
      <= (evaluateASTNode(node->getChild(1), m)));
    break;

  case AST_RELATIONAL_LT :
    result = (double) ((evaluateASTNode(node->getChild(0), m)) 
      < (evaluateASTNode(node->getChild(1), m)));
    break;

  case AST_RELATIONAL_NEQ :
    result = (double) ((evaluateASTNode(node->getChild(0), m)) 
      != (evaluateASTNode(node->getChild(1), m)));
    break;

  default:
    result = numeric_limits<double>::quiet_NaN();
    break;
  }


  return result;
}


bool 
SBMLTransforms::expandInitialAssignments(Model * m)
{
  IdList idsNoValues = mapComponentValues(m);
  IdList idsWithValues;

  IdValueIter iter;
  bool needToBail = false;

  do
  {
    /* need a fail safe in case a value is just missing */
    unsigned int num = m->getNumInitialAssignments();
    unsigned int count = num;
    
    /* list ids that have a calculated/assigned value */
    idsWithValues.clear();
    for (iter = mValues.begin(); iter != mValues.end(); iter++)
    {
      if (((*iter).second).second)
      {
        idsWithValues.append((*iter).first);
      }
    }

    for (unsigned int i = 0; i < m->getNumInitialAssignments(); i++)
    {
      if (!nodeContainsId(m->getInitialAssignment(i)->getMath(), idsNoValues))
      {
        if (!nodeContainsNameNotInList(m->getInitialAssignment(i)->getMath(), 
                                                                 idsWithValues))
        {
          std::string id = m->getInitialAssignment(i)->getSymbol();
          if (m->getCompartment(id) != NULL) 
          {
            if (expandInitialAssignment(m->getCompartment(id), 
                                        m->getInitialAssignment(i)))
            {
              m->removeInitialAssignment(id);
              count--;
            }
          }
          else if (m->getParameter(id) != NULL)
          {
            if (expandInitialAssignment(m->getParameter(id), 
                                        m->getInitialAssignment(i)))
            {
              m->removeInitialAssignment(id);
              count--;
            }
          }
          else if (m->getSpecies(id) != NULL)
          {
            if (expandInitialAssignment(m->getSpecies(id), 
                                        m->getInitialAssignment(i)))
            {
              m->removeInitialAssignment(id);
              count--;
            }
          }
          else 
          {
            for (unsigned int j = 0; j < m->getNumReactions(); j++)
            {
              Reaction * r = m->getReaction(j);
              unsigned int k;
              for (k = 0; k < r->getNumProducts(); k++)
              {
                if (r->getProduct(k)->getId() == id)
                {
                  if (expandInitialAssignment(r->getProduct(k), 
                                              m->getInitialAssignment(i)))
                  {
                    m->removeInitialAssignment(id);
                    count--;
                  }
                }
              }
              for (k = 0; k < r->getNumProducts(); k++)
              {
                if (r->getReactant(k)->getId() == id)
                {
                  if (expandInitialAssignment(r->getReactant(k), 
                                              m->getInitialAssignment(i)))
                  {
                    m->removeInitialAssignment(id);
                    count--;
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        needToBail = true;
      }
    }
    /* if count is still same nothing changed so bail or endlessly loop */
    if (count == num)
    {
      needToBail = true;
    }
  }
  while(m->getNumInitialAssignments() > 0 && needToBail == false);

  // clear the internal map of values
  mValues.clear();

  return true;
}


bool 
SBMLTransforms::expandInitialAssignment(Compartment * c, 
    const InitialAssignment *ia)
{
  bool success = false; 
  double value = evaluateASTNode(ia->getMath(), c->getModel());
  if (!isnan(value))
  {
    c->setSize(value);
    IdValueIter it = mValues.find(c->getId());
    ((*it).second).first = value;
    ((*it).second).second = true;
    success = true;
  }

  return success;
}

bool 
SBMLTransforms::expandInitialAssignment(Parameter * p, 
    const InitialAssignment *ia)
{
  bool success = false; 
  double value = evaluateASTNode(ia->getMath(), p->getModel());
  if (!isnan(value))
  {
    p->setValue(value);
    IdValueIter it = mValues.find(p->getId());
    ((*it).second).first = value;
    ((*it).second).second = true;
    success = true;
  }

  return success;
}

bool 
SBMLTransforms::expandInitialAssignment(SpeciesReference * sr, 
    const InitialAssignment *ia)
{
  bool success = false; 
  double value = evaluateASTNode(ia->getMath(), sr->getModel());
  if (!isnan(value))
  {
    sr->setStoichiometry(value);
    IdValueIter it = mValues.find(sr->getId());
    ((*it).second).first = value;
    ((*it).second).second = true;
    success = true;
  }

  return success;
}

bool 
SBMLTransforms::expandInitialAssignment(Species * s, 
    const InitialAssignment *ia)
{
  bool success = false; 
  double value = evaluateASTNode(ia->getMath(), s->getModel());
  if (!isnan(value))
  {
    if (s->getHasOnlySubstanceUnits())
    {
      s->setInitialAmount(value);
    }
    else
    {
      s->setInitialConcentration(value);
    }

    IdValueIter it = mValues.find(s->getId());
    ((*it).second).first = value;
    ((*it).second).second = true;
    success = true;
  }

  return success;
}

/** @endcond */

/** @cond doxygen-c-only */


LIBSBML_CPP_NAMESPACE_END

/** @endcond */

