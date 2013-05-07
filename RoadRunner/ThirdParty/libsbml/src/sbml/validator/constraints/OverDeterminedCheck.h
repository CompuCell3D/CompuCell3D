/**
 * @cond doxygen-libsbml-internal
 *
 * @file    OverDeterminedCheck.h
 * @brief   Checks for over determined models.
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

#ifndef OverDeterminedCheck_h
#define OverDeterminedCheck_h


#ifdef __cplusplus



#include <string>
#include <vector>

#include <algorithm>
#include <functional>

#include <map>

#include <sbml/validator/VConstraint.h>
#include "IdList.h"

LIBSBML_CPP_NAMESPACE_BEGIN

typedef std::map< std::string, IdList> graph;

class Model;
class Validator;


class OverDeterminedCheck: public TConstraint<Model>
{
public:

  /**
   * Creates a new Constraint with the given id.
   */
  OverDeterminedCheck (unsigned int id, Validator& v);

  /**
   * Destroys this Constraint.
   */
  virtual ~OverDeterminedCheck ();


protected:

  /**
   * Checks that a model is not over determined
   */
  virtual void check_ (const Model& m, const Model& object);

  /** 
   * creates equation vertexes according to the L2V2 spec 4.11.5 for every
   * 1. a Species structure that has the boundaryCondition field set to false 
   * and constant field set to false and which is referenced by one or more 
   * reactant or product lists of a Reaction structure containing a KineticLaw structure
   * 2. a Rule structure
   * 3. a KineticLaw structure
   */
  void writeEquationVertexes(const Model &);

  /**
   * creates variable vertexes according to the L2V2 spec 4.11.5 for
   * (a) every Species, Compartment and Parameter structure which has the
   * Constant field set to false; and 
   * (b) for every Reaction structure.
   */
  void writeVariableVertexes(const Model &);

  /**
   * creates a bipartite graph according to the L2V2 spec 4.11.5 
   * creates edges between the equation vertexes and the variable vertexes
   * graph produced is an id representimg the equation and an IdList
   * listing the edges the equation vertex is connected to
   */
  void createGraph(const Model &);

  /**
   * finds a maximal matching of the bipartite graph
   * adapted from the only implementation I could find:
   * # Hopcroft-Karp bipartite max-cardinality mMatching and max independent set
   * # David Eppstein, UC Irvine, 27 Apr 2002 - Python Cookbook
   *
   * returns an IdList of any equation vertexes that are unconnected 
   * in the maximal matching
   */ 
  IdList findMatching();

  /**
  * function that looks for alternative paths and adds these to the matching
  * where necessary
  */
  unsigned int Recurse(std::string);

  /**
   * Logs a message about overdetermined model.
   * As yet this only reports the problem - it doesnt really give
   * any additional information
   */
  void logOverDetermined (const Model &, const IdList& unmatched);


  IdList mEquations; // list of equation vertexes
  IdList mVariables; // list of variable vertexes
  graph mGraph;

  /* these are to enable the bipartite matching without passing variables */
  graph mMatching;
  graph mVarNeighInPrev;
  graph mEqnNeighInPrev;

  graph revisited;
  IdList visited;

};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */
#endif  /* OverDeterminedCheck_h */

/** @endcond */
