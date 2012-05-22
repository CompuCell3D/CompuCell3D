/*
  Last changed Time-stamp: <2007-10-26 17:39:56 raim>
  $Id: drawGraph.c,v 1.30 2008/11/07 08:51:53 raimc Exp $
*/
/* 
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. The software and
 * documentation provided hereunder is on an "as is" basis, and the
 * authors have no obligations to provide maintenance, support,
 * updates, enhancements or modifications.  In no event shall the
 * authors be liable to any party for direct, indirect, special,
 * incidental or consequential damages, including lost profits, arising
 * out of the use of this software and its documentation, even if the
 * authors have been advised of the possibility of such damage.  See
 * the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 *
 * The original code contained here was initially developed by:
 *
 *     Rainer Machne
 *
 * Contributor(s):
 *     Christoph Flamm
 */
/*! \defgroup drawGraph Graph Drawing
    \brief This optional module contains all functions to draw SBML and ODE
    Model structures as a graph with diverse output formats.

    Graph drawing is based on the Graphviz library. Unfortunately most
    graphviz versions currently have memory leaks. If you know, how these
    could be avoided, please contact us.
*/
/*@{*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Header Files for libsbml */
#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>

/* System specific definitions,
   created by configure script */
#ifndef WIN32
#include "config.h"
#endif

/* own header files */
#include "sbmlsolver/cvodeData.h"
#include "sbmlsolver/processAST.h"
#include "sbmlsolver/drawGraph.h"
#include "sbmlsolver/solverError.h"
#include "sbmlsolver/util.h"

/* Header Files for Graphviz */
#if USE_GRAPHVIZ
#if (GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 4) || GRAPHVIZ_MAJOR_VERSION >= 3
#include <gvc.h>
#else
#include <dotneato.h>
#include <gvrender.h>
#endif
#else
static int drawSensitivityTxt(cvodeData_t *data, char *file, double);
static int drawJacobyTxt(cvodeData_t *data, char *file);
static int drawModelTxt(Model_t *m, char *file);
#endif

#define WORDSIZE 10000

/** Draws a graph of the non-zero entries in the Jacobian matrix J =
    df/dx of an ODE system f(x,p,t) = dx/dt at the end-time of the
    last integration.

    Negative entries f(x1)/dx2 will be drawn as a red edge with a
    `tee' arrowhead from x2 to x1, representing the negative influence
    of x2 on x1. Positive entries will be drawn in black with a
    normal arrowhead. The edges are labelled by the actual value of
    the entry at time t. Note, that edge labels and also graph
    structure can change with time.

    The input cvodeData can be retrieved from an integratorInstance with
    IntegratorInstance_getData(). The output graph will be written to
    a file named `file.`format', where format can be e.g. `ps', `svg',
    `jpg', `png', `cmapx' etc. The latter retrieves an HTML image map.
    Please see the graphviz documentation for other available formats.
*/

SBML_ODESOLVER_API int drawJacoby(cvodeData_t *data, char *file, char *format)
{
  /** if SOSlib has been compiled without graphviz, the graph will be
      written to a text file in graphviz' dot format */
#if !USE_GRAPHVIZ

  SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_NO_GRAPHVIZ,
		    "odeSolver has been compiled without GRAPHIZ. ",
		    "Graphs are printed to stdout in graphviz' .dot format.");

  drawJacobyTxt(data, file);

#else

  int i, j;
  GVC_t *gvc;
  Agraph_t *g;
  Agnode_t *r;
  Agnode_t *s;  
  Agedge_t *e;
  Agsym_t *a;
  char name[WORDSIZE];
  char label[WORDSIZE];  
  char *output[4];
  char *command = "dot";
  char *formatopt;
  char *outfile;
  

  /* setting name of outfile */
  ASSIGN_NEW_MEMORY_BLOCK(outfile, strlen(file)+ strlen(format)+7, char, 0);
  sprintf(outfile, "-o%s_jm.%s", file, format);
  
  /* setting output format */
  ASSIGN_NEW_MEMORY_BLOCK(formatopt, strlen(format)+3, char, 0);
  sprintf(formatopt, "-T%s", format); 

  /* construct command-line */
  output[0] = command;
  output[1] = formatopt;
  output[2] = outfile;
  output[3] = NULL;
    
  /* set up renderer context */
  gvc = (GVC_t *) gvContext();
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION < 4
  dotneato_initialize(gvc, 3, output);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  parse_args(gvc, 3, output);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvParseArgs(gvc, 3, output);
#endif

  g = agopen("G", AGDIGRAPH);

  /* avoid overlapping nodes, for graph embedding by neato */
  a = agraphattr(g, "overlap", "");
  agxset(g, a->index, "scale");

  /* set graph label */
  if ( Model_isSetName(data->model->m) )
    sprintf(label, "%s at time %g",  Model_getName(data->model->m),
	    data->currenttime);
  else if ( Model_isSetId(data->model->m) )
    sprintf(label, "%s at time %g",  Model_getId(data->model->m),
	    data->currenttime);
  else
    sprintf(label, "label=\"at time %g\";\n", data->currenttime);

  a = agraphattr(g, "label", "");
  agxset(g, a->index, label);
  
  /*
    Set edges from species A to species B if the
    corresponding entry in the jacobian ((d[B]/dt)/d[A])
    is not '0'. Set edge color 'red' and arrowhead 'tee'
    if negative.
  */

  for ( i=0; i<data->model->neq; i++ )
  {
    for ( j=0; j<data->model->neq; j++ )
    {
      if ( evaluateAST(data->model->jacob[i][j], data) != 0 )
      {	
	sprintf(name, "%s", data->model->names[j]);
	r = agnode(g,name);
	agset(r, "label", data->model->names[j]);

	sprintf(label, "%s.htm", data->model->names[j]);
	a = agnodeattr(g, "URL", "");
	agxset(r, a->index, label);
	
	sprintf(name,"%s", data->model->names[i]);
	s = agnode(g,name);
	agset(s, "label", data->model->names[i]);

	sprintf(label, "%s.htm", data->model->names[i]);	
	a = agnodeattr(g, "URL", "");
	agxset(s, a->index, label);
	
	e = agedge(g,r,s);

	a = agedgeattr(g, "label", "");
	sprintf(name, "%g",  evaluateAST(data->model->jacob[i][j], data)); 
	agxset (e, a->index, name);
	
	if ( evaluateAST(data->model->jacob[i][j], data) < 0 )
	{
	  a = agedgeattr(g, "arrowhead", "");
	  agxset(e, a->index, "tee");
	  a = agedgeattr(g, "color", "");
	  agxset(e, a->index, "red"); 	    
	}	
      }
    }
  }
  
  /* Compute a layout */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  gvBindContext(gvc, g);
  dot_layout(g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  gvlayout_layout(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvLayoutJobs(gvc, g);
#endif
  
  /* Write the graph according to -T and -o options */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dotneato_write(gvc);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  emit_jobs(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvRenderJobs(gvc, g);
#endif
  
  /* Clean out layout data */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dot_cleanup(g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  gvlayout_cleanup(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvFreeLayout(gvc, g);
#endif
  
  /* Free graph structures */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dot_cleanup(g);
#endif
  agclose(g);

  /* Clean up output file and errors */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  gvFREEcontext(gvc);
  dotneato_eof(gvc);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  dotneato_terminate(gvc);
#elif (GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6) || GRAPHVIZ_MAJOR_VERSION >= 3
  gvFreeContext(gvc);
#endif

  xfree(formatopt);
  xfree(outfile);

#endif

  return 1;
}

#if !USE_GRAPHVIZ

static int drawJacobyTxt(cvodeData_t *data, char *file)
{

  int i, j;
  char filename[WORDSIZE];
  FILE *f;

  sprintf(filename, "%s.dot", file);
  f = fopen(filename, "w");
  fprintf(f ,"digraph jacoby {\n");
  fprintf(f ,"overlap=scale;\n");
  if ( Model_isSetName(data->model->m) )
    fprintf(f ,"label=\"%s at time %g\";\n", Model_getName(data->model->m),
	    data->currenttime);
  else if ( Model_isSetId(data->model->m) )
    fprintf(f ,"label=\"%s at time %g\";\n", Model_getId(data->model->m),
	    data->currenttime);
  else
    fprintf(f ,"label=\"at time %g\";\n", data->currenttime);


  /*
    Set edges from species A to species B if the
    corresponding entry in the jacobian ((d[B]/dt)/d[A])
    is not '0'. Set edge color 'red' and arrowhead 'tee'
    if negative.
  */


  for ( i=0; i<data->model->neq; i++ )
  {
    for ( j=0; j<data->model->neq; j++ )
    {
      if ( evaluateAST(data->model->jacob[i][j], data) != 0 )
      {
	fprintf(f ,"%s->%s [label=\"%g\" ",
		data->model->names[j],
		data->model->names[i],
		evaluateAST(data->model->jacob[i][j],
			    data));
	if ( evaluateAST(data->model->jacob[i][j], data) < 0 )
	  fprintf(f ,"arrowhead=tee color=red];\n");
	else
	  fprintf(f ,"];\n");
      }
    }
  }
  for ( i=0; i<data->model->neq; i++ )
  {
    fprintf(f ,"%s [label=\"%s\"];", data->model->names[i],
	    data->model->names[i]);
  }   
  fprintf(f, "}\n");
  return 1;
}

#endif


/** Draws a graph of the non-zero entries in the sensitivity matrix P
    = df/dp of an ODE system f(x,p,t) = dx/dt at the end-time of the
    last integration.

    Negative entries will f(x)/dp will be drawn as a red edge with a
    `tee' arrowhead from p to x, representing the negative influence
    of p on x.  Positive entries will be drawn in black with a normal
    arrowhead. The edges are labelled by the actual value of the entry
    at time t. Note, that edge labels and also graph structure can
    change with time.
    
    As this graph is usually of very high or full connectivity, i.e.
    no variable is completely independent of some parameter of the
    system, the function takes a threshold between 0 and 1 as an
    additional input. Only edges for entries greater then the
    threshold multiplied by the maximum entry for species x will be
    drawn.

    The input cvodeData can be retrieved from an integratorInstance with
    IntegratorInstance_getData(). The output graph will be written to
    a file named `file.`format', where format can be e.g. `ps', `svg',
    `jpg', `png', `cmapx' etc. The latter retrieves an HTML image map.
    Please see the graphviz documentation for other available formats.
*/

SBML_ODESOLVER_API int drawSensitivity(cvodeData_t *data, char *file, char *format, double threshold)
{

  /** if SOSlib has been compiled without graphviz, the graph will be
      written to a text file in graphviz' dot format */

#if !USE_GRAPHVIZ

  SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_NO_GRAPHVIZ,
		    "odeSolver has been compiled without GRAPHIZ. ",
		    "Graphs are printed to stdout in graphviz' .dot format.");

  drawSensitivityTxt(data, file, threshold);

#else

  int i, j;
  GVC_t *gvc;
  Agraph_t *g;
  Agnode_t *r;
  Agnode_t *s;  
  Agedge_t *e;
  Agsym_t *a;
  char name[WORDSIZE];
  char label[WORDSIZE];  
  char *output[4];
  char *command = "dot";
  char *formatopt;
  char *outfile;
  odeModel_t *om;
  odeSense_t *os;
  double *highest;
  double *lowest;

  om = data->model;
  os = data->os;

  /* setting name of outfile */
  ASSIGN_NEW_MEMORY_BLOCK(outfile, strlen(file)+ strlen(format)+7, char, 0);
  sprintf(outfile, "-o%s_s.%s", file, format);
  
  /* setting output format */
  ASSIGN_NEW_MEMORY_BLOCK(formatopt, strlen(format)+3, char, 0);
  sprintf(formatopt, "-T%s", format); 

  /* construct command-line */
  output[0] = command;
  output[1] = formatopt;
  output[2] = outfile;
  output[3] = NULL;
    
  /* set up renderer context */
  gvc = (GVC_t *) gvContext();
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION < 4
  dotneato_initialize(gvc, 3, output);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  parse_args(gvc, 3, output);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvParseArgs(gvc, 3, output);
#endif

  g = agopen("G", AGDIGRAPH);

  /* avoid overlapping nodes, for graph embedding by neato */
  a = agraphattr(g, "overlap", "");
  agxset(g, a->index, "scale");

  /* set graph label */
  if ( Model_isSetName(om->m) )
    sprintf(label, "%s at time %g", Model_getName(om->m), data->currenttime);
  else if ( Model_isSetId(om->m) )
    sprintf(label, "%s at time %g", Model_getId(om->m), data->currenttime);
  else
    sprintf(label, "label=\"at time %g\";\n", data->currenttime);

  a = agraphattr(g, "label", "");
  agxset(g, a->index, label);
  
  /*
    Set edges from species A to species B if the
    corresponding entry in the jacobian ((d[B]/dt)/d[A])
    is not '0'. Set edge color 'red' and arrowhead 'tee'
    if negative.
  */
  ASSIGN_NEW_MEMORY_BLOCK(highest, data->nsens, double, 0);  
  ASSIGN_NEW_MEMORY_BLOCK(lowest, data->nsens, double, 0);
  
  for ( j=0; j<data->nsens; j++ )
  {
    highest[j] = 0;
    lowest[j] = 0;
    for ( i=0; i<om->neq; i++ )
    {
      if ( data->sensitivity[i][j] > highest[j] )
	highest[j] = data->sensitivity[i][j];
      if ( data->sensitivity[i][j] < lowest[j] )
	lowest[j] = data->sensitivity[i][j];
    }
  }

  for ( i=0; i<om->neq; i++ )
  {
    for ( j=0; j<data->nsens; j++ )
    {
      if ( (data->sensitivity[i][j] > threshold*highest[j]) ||
	   (data->sensitivity[i][j] < threshold*lowest[j]) )
      {	
	sprintf(name, "%s", om->names[os->index_sens[j]]);
	r = agnode(g,name);
	agset(r, "label", om->names[os->index_sens[j]]);

	sprintf(label, "%s.htm", om->names[os->index_sens[j]]);
	a = agnodeattr(g, "URL", "");
	agxset(r, a->index, label);
	
	sprintf(name,"%s", om->names[i]);
	s = agnode(g,name);
	agset(s, "label", om->names[i]);

	sprintf(label, "%s.htm", om->names[i]);	
	a = agnodeattr(g, "URL", "");
	agxset(s, a->index, label);
	
	e = agedge(g,r,s);

	a = agedgeattr(g, "label", "");
	sprintf(name, "%g",  data->sensitivity[i][j]); 
	agxset (e, a->index, name);


	
	if ( data->sensitivity[i][j] < 0 )
	{
	  a = agedgeattr(g, "arrowhead", "");
	  agxset(e, a->index, "tee");
	  a = agedgeattr(g, "color", "");
	  agxset(e, a->index, "red"); 	    
	}	
      }
    }
  }
  
  /* Compute a layout */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  gvBindContext(gvc, g);
  dot_layout(g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  gvlayout_layout(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvLayoutJobs(gvc, g);
#endif
  
  /* Write the graph according to -T and -o options */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dotneato_write(gvc);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  emit_jobs(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvRenderJobs(gvc, g);
#endif
  
  /* Clean out layout data */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dot_cleanup(g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  gvlayout_cleanup(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvFreeLayout(gvc, g);
#endif
  
  /* Free graph structures */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dot_cleanup(g);
#endif
  agclose(g);

  /* Clean up output file and errors */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  gvFREEcontext(gvc);
  dotneato_eof(gvc);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  dotneato_terminate(gvc);
#elif (GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6) || GRAPHVIZ_MAJOR_VERSION >= 3
  gvFreeContext(gvc);
#endif

  xfree(formatopt);
  xfree(outfile);
  free(highest);
  free(lowest);
#endif

  return 1;
}

#if !USE_GRAPHVIZ

static int drawSensitivityTxt(cvodeData_t *data, char *file, double threshold)
{

  int i, j;
  char filename[WORDSIZE];
  FILE *f;
  odeModel_t *om;
  odeSense_t *os;
  double *highest;
  double *lowest;
  
  om = data->model;
  os = data->os;
  
  sprintf(filename, "%s.dot", file);
  f = fopen(filename, "w");
  fprintf(f ,"digraph jacoby {\n");
  fprintf(f ,"overlap=scale;\n");
  if ( Model_isSetName(om->m) )
    fprintf(f ,"label=\"%s at time %g\";\n", Model_getName(om->m),
	    data->currenttime);
  else if ( Model_isSetId(om->m) )
    fprintf(f ,"label=\"%s at time %g\";\n", Model_getId(om->m),
	    data->currenttime);
  else
    fprintf(f ,"label=\"at time %g\";\n", data->currenttime);


  /*
    Set edges from species A to species B if the
    corresponding entry in the jacobian ((d[B]/dt)/d[A])
    is not '0'. Set edge color 'red' and arrowhead 'tee'
    if negative.
  */

  ASSIGN_NEW_MEMORY_BLOCK(highest, data->nsens, double, 0);  
  ASSIGN_NEW_MEMORY_BLOCK(lowest, data->nsens, double, 0);
  
  for ( j=0; j<data->nsens; j++ )
  {
    highest[j] = 0;
    lowest[j] = 0;
    for ( i=0; i<om->neq; i++ )
    {
      if ( data->sensitivity[i][j] > highest[j] )
	highest[j] = data->sensitivity[i][j];
      if ( data->sensitivity[i][j] < lowest[j] )
	lowest[j] = data->sensitivity[i][j];
    }
  }
  
  for ( i=0; i<om->neq; i++ )
  {
    for ( j=0; j<data->nsens; j++ )
    {
      if ( (data->sensitivity[i][j] > threshold*highest[j]) ||
	   (data->sensitivity[i][j] < threshold*lowest[j]) )
      {	
	fprintf(f ,"%s->%s [label=\"%g\" ",
		om->names[os->index_sens[j]],
		om->names[i],
		data->sensitivity[i][j]);
	
	if ( data->sensitivity[i][j] < 0 ) 
	  fprintf(f ,"arrowhead=tee color=red];\n");
	else 
	  fprintf(f ,"];\n");
      }
    }
  }
  for ( i=0; i<om->neq; i++ )
    fprintf(f ,"%s [label=\"%s\"];", om->names[i], om->names[i]);
  
  for ( i=0; i<data->nsens; i++ ) 
    fprintf(f ,"%s [label=\"%s\"];", om->names[os->index_sens[i]],
	    om->names[os->index_sens[i]]);

  fprintf(f, "}\n");
  free(highest);
  free(lowest);
  return 1;
}

#endif





/** Draws a bipartite graph of the reaction network in the passed SBML
    model `m' (as libSBML Model_t).

    Reactions will be boxes and species will be ellipses. The
    stoichiometry will be drawn as edge labels, if other than
    1. Boundary species will be drawn in blue and constant species
    will be drawn in green where the latter overrides the former.
    Reaction modifier interactions will be drawn as dashed arrows with
    an open circle as an arrowhead.

    The output graph will be written to a file named `file.`format',
    where format can be e.g. `ps', `svg', `jpg', `png', `cmapx'
    etc. The latter retrieves an HTML image map.  Please see the
    graphviz documentation for other available formats.
*/

SBML_ODESOLVER_API int drawModel(Model_t *m, char* file, char *format)
{
  
  /** if SOSlib has been compiled without graphviz, the graph will be
      written to a text file in graphviz' dot format */  
#if !USE_GRAPHVIZ

  SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_NO_GRAPHVIZ,
		    "odeSolver has been compiled without GRAPHIZ. ",
		    "Graphs are printed to stdout in graphviz' .dot format.");
  drawModelTxt(m, file);
  
#else

  GVC_t *gvc;
  Agraph_t *g;
  Agnode_t *r;
  Agnode_t *s;  
  Agedge_t *e;
  Agsym_t *a;
  Species_t *sp;
  Reaction_t *re;
  const ASTNode_t *math;  
  SpeciesReference_t *sref;
  SpeciesReference_t *mref; 
  char *output[4];
  char *command = "dot";
  char *formatopt;
  char *outfile;
  int i,j;
  int reversible;
  char name[WORDSIZE];
  char label[WORDSIZE];

  /* setting name of outfile */
  ASSIGN_NEW_MEMORY_BLOCK(outfile, strlen(file)+ strlen(format)+7, char, 0);
  sprintf(outfile, "-o%s_rn.%s", file, format);

  /* setting output format */
  ASSIGN_NEW_MEMORY_BLOCK(formatopt, strlen(format)+3, char, 0);
  sprintf(formatopt, "-T%s", format);

  /* construct command-line */
  output[0] = command;
  output[1] = formatopt;
  output[2] = outfile;
  output[3] = NULL;

  /* set up renderer context */
  gvc = (GVC_t *) gvContext();
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION < 4
  dotneato_initialize(gvc, 3, output);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  parse_args(gvc, 3, output);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvParseArgs(gvc, 3, output);  
#endif  

  g = agopen("G", AGDIGRAPH);
  
  /* avoid overlapping nodes, for graph embedding by neato */ 
  a = agraphattr(g, "overlap", "");
  agxset(g, a->index, "scale");

  for ( i=0; i<Model_getNumReactions(m); i++ )
  {
    re = Model_getReaction(m,i);
    reversible = Reaction_getReversible(re);
    sprintf(name, "%s", Reaction_getId(re));
    r = agnode(g,name);
    a = agnodeattr(g, "shape", "ellipse");    
    agxset(r, a->index, "box");

    if ( Reaction_getFast(re) )
    {
      a = agnodeattr(g, "color", "");
      agxset(r, a->index, "red");
      a = agnodeattr(g, "fontcolor", "");
      agxset(r, a->index, "red");
    }
    
    sprintf(label, "%s", Reaction_isSetName(re) ?
	    Reaction_getName(re) : Reaction_getId(re));
    agset(r, "label", label);
    
    sprintf(label, "%s.htm", Reaction_getId(re));
    a = agnodeattr(g, "URL", "");
    agxset(r, a->index, label);
    
    for ( j=0; j<Reaction_getNumModifiers(re); j++ )
    {
      mref = Reaction_getModifier(re,j);
      sp = Model_getSpeciesById(m, SpeciesReference_getSpecies(mref));
      
      sprintf(name,"%s", Species_getId(sp));
      s = agnode(g,name);
      sprintf(label, "%s", Species_isSetName(sp) ? 
	      Species_getName(sp) : Species_getId(sp));
      agset(s, "label", label);

      if ( Species_getBoundaryCondition(sp) )
      {
	a = agnodeattr(g, "color", "");
	agxset(s, a->index, "blue");
      }
      if ( Species_getConstant(sp) )
      {
	a = agnodeattr(g, "color", "");
	agxset(s, a->index, "green4");
      }

      sprintf(label, "%s.htm", Species_getId(sp));
      a = agnodeattr(g, "URL", "");
      agxset(s, a->index, label);
	
      e = agedge(g,s,r);
      a = agedgeattr(g, "style", "");
      agxset(e, a->index, "dashed");
      a = agedgeattr(g, "arrowhead", "");
      agxset(e, a->index, "odot");
    }

    for ( j=0; j<Reaction_getNumReactants(re); j++ )
    {
      sref = Reaction_getReactant(re,j);
      sp = Model_getSpeciesById(m, SpeciesReference_getSpecies(sref));
      
      sprintf(name,"%s", Species_getId(sp));
      s = agnode(g, name);
      sprintf(label, "%s", Species_isSetName(sp) ? 
	      Species_getName(sp) : Species_getId(sp));
      agset(s, "label", label);

      if ( Species_getBoundaryCondition(sp) )
      {
	a = agnodeattr(g, "color", "");
	agxset(s, a->index, "blue");
      }
      if ( Species_getConstant(sp) )
      {
	a = agnodeattr(g, "color", "");
	agxset(s, a->index, "green4");
      }

      sprintf(label, "%s.htm", Species_getId(sp));
      a = agnodeattr(g, "URL", "");
      agxset(s, a->index, label);
      
      e = agedge(g,s,r);      
      a = agedgeattr(g, "label", "");
      
      if ( (SpeciesReference_isSetStoichiometryMath(sref)) )
      {
	math = StoichiometryMath_getMath(SpeciesReference_getStoichiometryMath(sref));
	if ( (strcmp(SBML_formulaToString(math),"1") != 0) ) 
	  agxset (e, a->index, SBML_formulaToString(math));	
      }
      else
	if ( SpeciesReference_getStoichiometry(sref) != 1 )
	{
	  sprintf(name, "%g", SpeciesReference_getStoichiometry(sref));
	  agxset (e, a->index, name);
	}      
      
      if ( reversible == 1 )
      {
	a = agedgeattr(g, "arrowhead", "");
	agxset(e, a->index, "diamond");
	a = agedgeattr(g, "arrowtail", "");
	agxset(e, a->index, "odiamond");
      }
    }
    
    for ( j=0; j<Reaction_getNumProducts(re); j++ )
    {
      sref = Reaction_getProduct(re,j);
      sp = Model_getSpeciesById(m, SpeciesReference_getSpecies(sref));
      sprintf(name,"%s", Species_getId(sp));
      s = agnode(g,name);
      sprintf(label, "%s", Species_isSetName(sp) ? 
	      Species_getName(sp) : Species_getId(sp));
      agset(s, "label", label);

      if ( Species_getBoundaryCondition(sp) )
      {
	a = agnodeattr(g, "color", "");
	agxset(s, a->index, "blue");
      }
      if ( Species_getConstant(sp) )
      {
	a = agnodeattr(g, "color", "");
	agxset(s, a->index, "green4");
      }

      sprintf(label, "%s.htm", Species_getId(sp));
      a = agnodeattr(g, "URL", "");
      agxset(s, a->index, label);
            
      e = agedge(g,r,s);
      a = agedgeattr(g, "label", "");
      
      if ( SpeciesReference_isSetStoichiometryMath(sref) )
      {
	math = StoichiometryMath_getMath(SpeciesReference_getStoichiometryMath(sref));
	if ( (strcmp(SBML_formulaToString(math),"1") != 0) ) 
	  agxset (e, a->index, SBML_formulaToString(math));	
      }
      else      
	if ( SpeciesReference_getStoichiometry(sref) != 1 )
	{
	  sprintf(name, "%g",SpeciesReference_getStoichiometry(sref));
	  agxset (e, a->index,name);
	}
      
      if ( reversible == 1 )
      {
	a = agedgeattr(g, "arrowhead", "");
	agxset(e, a->index, "diamond");
	a = agedgeattr(g, "arrowtail", "");
	agxset(e, a->index, "odiamond");
      }
    }   
  }

  /* Compute a layout */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  gvBindContext(gvc, g);
  dot_layout(g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  gvlayout_layout(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvLayoutJobs(gvc, g);
#endif

  /* Write the graph according to -T and -o options */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dotneato_write(gvc);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  emit_jobs(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvRenderJobs(gvc, g);
#endif
  
  /* Clean out layout data */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dot_cleanup(g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  gvlayout_cleanup(gvc, g);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvFreeLayout(gvc, g);
#endif
  
  /* Free graph structures */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  dot_cleanup(g);
#else
  agclose(g);
#endif

  /* Clean up output file and errors */
#if GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION <= 2
  gvFREEcontext(gvc);
  dotneato_eof(gvc);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION == 4
  dotneato_terminate(gvc);
#elif GRAPHVIZ_MAJOR_VERSION == 2 && GRAPHVIZ_MINOR_VERSION >= 6 || GRAPHVIZ_MAJOR_VERSION >= 3
  gvFreeContext(gvc); 
#endif  

  xfree(formatopt);  
  xfree(outfile);
  
#endif
  
  return 1;

}

#if !USE_GRAPHVIZ

static int drawModelTxt(Model_t *m, char *file)
{
  Species_t *s;
  Reaction_t *re;
  const ASTNode_t *math;
  SpeciesReference_t *sref;
  SpeciesReference_t *mref;
  int i,j;
  int reversible;
  char filename[WORDSIZE];
  FILE *f;
  
  sprintf(filename, "%s.dot", file);
  f = fopen(filename, "w");

  fprintf(f ,"digraph reactionnetwork {\n");
  fprintf(f ,"label=\"%s\";\n",
	  Model_isSetName(m) ?
	  Model_getName(m) : (Model_isSetId(m) ? Model_getId(m) : "noId") );
  fprintf(f ,"overlap=scale;\n");
 
  for ( i=0; i<Model_getNumReactions(m); i++ )
  {    
    re = Model_getReaction(m,i);
    reversible = Reaction_getReversible(re);
    
    for ( j=0; j<Reaction_getNumModifiers(re); j++ )
    {
      mref = Reaction_getModifier(re,j);
      fprintf(f ,"%s->%s [style=dashed arrowhead=odot];\n",
	      SpeciesReference_getSpecies(mref), Reaction_getId(re));
    }

    for ( j=0; j<Reaction_getNumReactants(re); j++ )
    {
      sref = Reaction_getReactant(re,j);
      fprintf(f ,"%s->%s [label=\"",
	      SpeciesReference_getSpecies(sref), Reaction_getId(re));
      
      if ( (SpeciesReference_isSetStoichiometryMath(sref)) )
      {
	math = SpeciesReference_getStoichiometryMath(sref);
	if ( (strcmp(SBML_formulaToString(math),"1") != 0) ) 
	  fprintf(f ,"%s", SBML_formulaToString(math));
      }
      else 
	if ( SpeciesReference_getStoichiometry(sref) != 1)
	  fprintf(f ,"%g",SpeciesReference_getStoichiometry(sref));
	
      
      if ( reversible == 1 ) 
	fprintf(f ,"\" arrowtail=onormal];\n");
      else 
	fprintf(f ,"\" ];\n");

    }

    for ( j=0; j<Reaction_getNumProducts(re); j++ )
    {
      sref = Reaction_getProduct(re,j);
      fprintf(f ,"%s->%s [label=\"",
	      Reaction_getId(re), SpeciesReference_getSpecies(sref));
      if ( (SpeciesReference_isSetStoichiometryMath(sref)) )
      {
	math = SpeciesReference_getStoichiometryMath(sref);
	if ( (strcmp(SBML_formulaToString(math),"1") != 0) ) 
	  fprintf(f ,"%s ", SBML_formulaToString(math));
	
      }
      else 
	if ( SpeciesReference_getStoichiometry(sref) != 1) 
	  fprintf(f ,"%g ",SpeciesReference_getStoichiometry(sref));	
      
      if ( reversible == 1 ) 
	fprintf(f ,"\" arrowtail=onormal];\n");
      else 
	fprintf(f ,"\" ];\n");

    }
    
  }
  
  for ( i=0; i<Model_getNumReactions(m); i++ )
  {
    re = Model_getReaction(m,i);
    fprintf(f ,"%s [label=\"%s\" shape=box];\n",
	    Reaction_getId(re),
	    Reaction_isSetName(re) ?
	    Reaction_getName(re) : Reaction_getId(re));
  }

  for ( i=0; i<Model_getNumSpecies(m); i++)
  {
    s = Model_getSpecies(m, i);
    fprintf(f ,"%s [label=\"%s\"];",
	    Species_getId(s),
	    Species_isSetName(s) ? Species_getName(s) : Species_getId(s));
  }  
  fprintf(f ,"}\n");
  return 1;
}

#endif
/** @} */
/* End of file */
