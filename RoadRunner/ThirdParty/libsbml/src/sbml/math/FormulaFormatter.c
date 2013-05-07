/**
 * @file    FormulaFormatter.c
 * @brief   Formats an AST formula tree as an SBML formula string.
 * @author  Ben Bornstein
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
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <sbml/common/common.h>
#include <sbml/math/FormulaFormatter.h>

#if defined(_MSC_VER) || defined(__BORLANDC__)
#  include <float.h>
#  define finite(d) _finite(d)
#  define isnan(d)  _isnan(d)
#endif


/** @cond doxygen-c-only */
/**
 * Converts an AST to a string representation of a formula using a syntax
 * basically derived from SBML Level&nbsp;1.
 *
 * @if clike The text-string form of mathematical formulas produced by
 * SBML_formulaToString() and read by SBML_parseFormula() and SBML_parseL3Formula() 
 * are in a C-inspired infix notation.  A formula in
 * this text-string form therefore can be handed to a program that
 * understands SBML mathematical expressions, or used as part
 * of a formula translation system.  The syntax is described in detail in
 * the documentation for ASTNode. @endif@if java The text-string form of
 * mathematical formulas produced by <code><a
 * href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode)">
 * libsbml.formulaToString()</a></code> and read by
 * <code><a href="libsbml.html#parseFormula(java.lang.String)">
 * libsbml.parseFormula()</a></code> and
 * <code><a href="libsbml.html#parseL3Formula(java.lang.String)">
 * libsbml.parseL3Formula()</a></code> are in a 
 * simple C-inspired infix notation.  A
 * formula in this text-string form therefore can be handed to a program
 * that understands SBML mathematical expressions, or used as
 * part of a formula translation system.  The syntax is described in detail
 * in the documentation for ASTNode.   @endif
 *
 * Note that this facility is provided as a convenience by libSBML&mdash;the
 * MathML standard does not actually define a "string-form" equivalent to
 * MathML expression trees, so the choice of formula syntax is somewhat
 * arbitrary.  The approach taken by libSBML is to use the syntax defined by
 * SBML Level&nbsp;1 (which in fact used a text-string representation of
 * formulas and not MathML).  This formula syntax is based mostly on C
 * programming syntax, and may contain operators, function calls, symbols,
 * and white space characters.  The following table provides the precedence
 * rules for the different entities that may appear in formula strings.
 *
 * @htmlinclude math-precedence-table.html
 * 
 * In the table above, @em operand implies the construct is an operand, @em
 * prefix implies the operation is applied to the following arguments, @em
 * unary implies there is one argument, and @em binary implies there are
 * two arguments.  The values in the <b>Precedence</b> column show how the
 * order of different types of operation are determined.  For example, the
 * expression <code>a * b + c</code> is evaluated as <code>(a * b) +
 * c</code> because the @c * operator has higher precedence.  The
 * <b>Associates</b> column shows how the order of similar precedence
 * operations is determined; for example, <code>a - b + c</code> is
 * evaluated as <code>(a - b) + c</code> because the @c + and @c -
 * operators are left-associative.
 *
 * The function call syntax consists of a function name, followed by optional
 * white space, followed by an opening parenthesis token, followed by a
 * sequence of zero or more arguments separated by commas (with each comma
 * optionally preceded and/or followed by zero or more white space
 * characters, followed by a closing parenthesis token.  The function name
 * must be chosen from one of the pre-defined functions in SBML or a
 * user-defined function in the model.  The following table lists the names
 * of certain common mathematical functions; this table corresponds to
 * Table&nbsp;6 in the <a target="_blank" href="http://sbml.org/Documents/Specifications#SBML_Level_1_Version_2">SBML Level&nbsp;1 Version&nbsp;2 specification</a>:
 *
 * @htmlinclude string-functions-table.html
 *
 * @warning There are differences between the symbols used to represent the
 * common mathematical functions and the corresponding MathML token names.
 * This is a potential source of incompatibilities.  Note in particular that
 * in this text-string syntax, <code>log(x)</code> represents the natural
 * logarithm, whereas in MathML, the natural logarithm is
 * <code>&lt;ln/&gt;</code>.  Application writers are urged to be careful
 * when translating between text forms and MathML forms, especially if they
 * provide a direct text-string input facility to users of their software
 * systems.<br><br>
 * @htmlinclude L1-math-syntax-warning.html
 *
 * @param tree the AST to be converted.
 * 
 * @return the formula from the given AST as an SBML Level 1 text-string
 * mathematical formula.  The caller owns the returned string and is
 * responsible for freeing it when it is no longer needed.
 *
 * @see SBML_parseFormula()
 * @see SBML_parseL3Formula()
 */
LIBSBML_EXTERN
char *
SBML_formulaToString (const ASTNode_t *tree)
{
  char           *s;

  if (tree == NULL)
  {
    s = NULL;
  }
  else
  {
    StringBuffer_t *sb = StringBuffer_create(128);

    FormulaFormatter_visit(NULL, tree, sb);
    s = StringBuffer_getBuffer(sb);
    safe_free(sb);
  }
  return s;
}
/** @endcond */


/**
 * @cond doxygen-libsbml-internal
 * The rest of this file is internal code.
 */


/**
 * @return true (non-zero) if the given ASTNode is to formatted as a
 * function.
 */
int
FormulaFormatter_isFunction (const ASTNode_t *node)
{
  return
    ASTNode_isFunction  (node) ||
    ASTNode_isLambda    (node) ||
    ASTNode_isLogical   (node) ||
    ASTNode_isRelational(node);
}


/**
 * @return true (non-zero) if the given child ASTNode should be grouped
 * (with parenthesis), false (0) otherwise.
 *
 * A node should be group if it is not an argument to a function and
 * either:
 *
 *   - The parent node has higher precedence than the child, or
 *
 *   - If parent node has equal precedence with the child and the child is
 *     to the right.  In this case, operator associativity and right-most
 *     AST derivation enforce the grouping.
 */
int
FormulaFormatter_isGrouped (const ASTNode_t *parent, const ASTNode_t *child)
{
  int pp, cp;
  int pt, ct;
  int group = 0;


  if (parent != NULL)
  {
    if (!FormulaFormatter_isFunction(parent))
    {
      pp = ASTNode_getPrecedence(parent);
      cp = ASTNode_getPrecedence(child);

      if (pp > cp)
      {
        group = 1;
      }
      else if (pp == cp)
      {
        /**
         * Group only if i) child is to the right and ii) both parent and
         * child are either not the same, or if they are the same, they
         * should be non-associative operators (i.e. AST_MINUS or
         * AST_DIVIDE).  That is, do not group a parent and right child
         * that are either both AST_PLUS or both AST_TIMES operators.
         */
        if (ASTNode_getRightChild(parent) == child)
        {
          pt = ASTNode_getType(parent);
          ct = ASTNode_getType(child);

          group = ((pt != ct) || (pt == AST_MINUS || pt == AST_DIVIDE));
        }
      }
    }
  }

  return group;
}


/**
 * Formats the given ASTNode as an SBML L1 token and appends the result to
 * the given StringBuffer.
 */
void
FormulaFormatter_format (StringBuffer_t *sb, const ASTNode_t *node)
{
  if (sb == NULL) return;
  if (ASTNode_isOperator(node))
  {
    FormulaFormatter_formatOperator(sb, node);
  }
  else if (ASTNode_isFunction(node))
  {
    FormulaFormatter_formatFunction(sb, node);
  }
  else if (ASTNode_isInteger(node))
  {
    StringBuffer_appendInt(sb, ASTNode_getInteger(node));
  }
  else if (ASTNode_isRational(node))
  {
    FormulaFormatter_formatRational(sb, node);
  }
  else if (ASTNode_isReal(node))
  {
    FormulaFormatter_formatReal(sb, node);
  }
  else if ( !ASTNode_isUnknown(node) )
  {
    StringBuffer_append(sb, ASTNode_getName(node));
  }
}


/**
 * Formats the given ASTNode as an SBML L1 function name and appends the
 * result to the given StringBuffer.
 */
void
FormulaFormatter_formatFunction (StringBuffer_t *sb, const ASTNode_t *node)
{
  ASTNodeType_t type = ASTNode_getType(node);


  switch (type)
  {
    case AST_FUNCTION_ARCCOS:
      StringBuffer_append(sb, "acos");
      break;

    case AST_FUNCTION_ARCSIN:
      StringBuffer_append(sb, "asin");
      break;

    case AST_FUNCTION_ARCTAN:
      StringBuffer_append(sb, "atan");
      break;

    case AST_FUNCTION_CEILING:
      StringBuffer_append(sb, "ceil");
      break;

    case AST_FUNCTION_LN:
      StringBuffer_append(sb, "log");
      break;

    case AST_FUNCTION_POWER:
      StringBuffer_append(sb, "pow");
      break;

    default:
      StringBuffer_append(sb, ASTNode_getName(node));
      break;
  }
}


/**
 * Formats the given ASTNode as an SBML L1 operator and appends the result
 * to the given StringBuffer.
 */
void
FormulaFormatter_formatOperator (StringBuffer_t *sb, const ASTNode_t *node)
{
  ASTNodeType_t type = ASTNode_getType(node);


  if (type != AST_POWER)
  {
    StringBuffer_appendChar(sb, ' ');
  }

  StringBuffer_appendChar(sb, ASTNode_getCharacter(node));

  if (type != AST_POWER)
  {
    StringBuffer_appendChar(sb, ' ');
  }
}


/**
 * Formats the given ASTNode as a rational number and appends the result to
 * the given StringBuffer.  For SBML L1 this amounts to:
 *
 *   "(numerator/denominator)"
 */
void
FormulaFormatter_formatRational (StringBuffer_t *sb, const ASTNode_t *node)
{
  StringBuffer_appendChar( sb, '(');
  StringBuffer_appendInt ( sb, ASTNode_getNumerator(node)   );
  StringBuffer_appendChar( sb, '/');
  StringBuffer_appendInt ( sb, ASTNode_getDenominator(node) );
  StringBuffer_appendChar( sb, ')');
}


/**
 * Formats the given ASTNode as a real number and appends the result to
 * the given StringBuffer.
 */
void
FormulaFormatter_formatReal (StringBuffer_t *sb, const ASTNode_t *node)
{
  double value = ASTNode_getReal(node);
  int    sign;


  if (isnan(value))
  {
    StringBuffer_append(sb, "NaN");
  }
  else if ((sign = util_isInf(value)) != 0)
  {
    if (sign == -1)
    {
      StringBuffer_appendChar(sb, '-');
    }

    StringBuffer_append(sb, "INF");
  }
  else if (util_isNegZero(value))
  {
    StringBuffer_append(sb, "-0");
  }
  else
  {
    if (ASTNode_getType(node) == AST_REAL_E)
    {
      StringBuffer_appendExp(sb, value);
    }
    else
    {
      StringBuffer_appendReal(sb, value);
    }
  }
}

int ASTNode_isPlus0(const ASTNode_t *node)
{
  return ASTNode_getType(node) == AST_PLUS && ASTNode_getNumChildren(node) == 0;
}

int ASTNode_isTimes0(const ASTNode_t *node)
{
  return ASTNode_getType(node) == AST_TIMES && ASTNode_getNumChildren(node) == 0;
}

int ASTNode_isTimes1(const ASTNode_t *node)
{
  return ASTNode_getType(node) == AST_TIMES && ASTNode_getNumChildren(node) == 1;
}

/**
 * Visits the given ASTNode node.  This function is really just a
 * dispatcher to either SBML_formulaToString_visitFunction() or
 * SBML_formulaToString_visitOther().
 */
void
FormulaFormatter_visit ( const ASTNode_t *parent,
                         const ASTNode_t *node,
                         StringBuffer_t  *sb )
{

  if (ASTNode_isLog10(node))
  {
    FormulaFormatter_visitLog10(parent, node, sb);
  }
  else if (ASTNode_isSqrt(node))
  {
    FormulaFormatter_visitSqrt(parent, node, sb);
  }
  else if (FormulaFormatter_isFunction(node))
  {
    FormulaFormatter_visitFunction(parent, node, sb);
  }
  else if (ASTNode_isUMinus(node))
  {
    FormulaFormatter_visitUMinus(parent, node, sb);
  }
  else if (ASTNode_isUPlus(node) || ASTNode_isTimes1(node))
  {
    FormulaFormatter_visit(node, ASTNode_getChild(node, 0), sb);
  }
  else if (ASTNode_isPlus0(node))
  {
    StringBuffer_appendInt(sb, 0);
  }
  else if (ASTNode_isTimes0(node))
  {
    StringBuffer_appendInt(sb, 1);
  }
  else
  {
    FormulaFormatter_visitOther(parent, node, sb);
  }
}


/**
 * Visits the given ASTNode as a function.  For this node only the
 * traversal is preorder.
 */
void
FormulaFormatter_visitFunction ( const ASTNode_t *parent,
                                 const ASTNode_t *node,
                                 StringBuffer_t  *sb )
{
  unsigned int numChildren = ASTNode_getNumChildren(node);
  unsigned int n;


  FormulaFormatter_format(sb, node);
  StringBuffer_appendChar(sb, '(');

  if (numChildren > 0)
  {
    FormulaFormatter_visit( node, ASTNode_getChild(node, 0), sb );
  }

  for (n = 1; n < numChildren; n++)
  {
    StringBuffer_appendChar(sb, ',');
    StringBuffer_appendChar(sb, ' ');
    FormulaFormatter_visit( node, ASTNode_getChild(node, n), sb );
  }

  StringBuffer_appendChar(sb, ')');
}


/**
 * Visits the given ASTNode as the function "log(10, x)" and in doing so,
 * formats it as "log10(x)" (where x is any subexpression).
 */
void
FormulaFormatter_visitLog10 ( const ASTNode_t *parent,
                              const ASTNode_t *node,
                              StringBuffer_t  *sb )
{
  StringBuffer_append(sb, "log10(");
  FormulaFormatter_visit(node, ASTNode_getChild(node, 1), sb);
  StringBuffer_appendChar(sb, ')');
}


/**
 * Visits the given ASTNode as the function "root(2, x)" and in doing so,
 * formats it as "sqrt(x)" (where x is any subexpression).
 */
void
FormulaFormatter_visitSqrt ( const ASTNode_t *parent,
                             const ASTNode_t *node,
                             StringBuffer_t  *sb )
{
  StringBuffer_append(sb, "sqrt(");
  FormulaFormatter_visit(node, ASTNode_getChild(node, 1), sb);
  StringBuffer_appendChar(sb, ')');
}


/**
 * Visits the given ASTNode as a unary minus.  For this node only the
 * traversal is preorder.
 */
void
FormulaFormatter_visitUMinus ( const ASTNode_t *parent,
                               const ASTNode_t *node,
                               StringBuffer_t  *sb )
{
  StringBuffer_appendChar(sb, '-');
  FormulaFormatter_visit ( node, ASTNode_getLeftChild(node), sb );
}


/**
 * Visits the given ASTNode and continues the inorder traversal.
 */
void
FormulaFormatter_visitOther ( const ASTNode_t *parent,
                              const ASTNode_t *node,
                              StringBuffer_t  *sb )
{
  unsigned int numChildren = ASTNode_getNumChildren(node);
  unsigned int group       = FormulaFormatter_isGrouped(parent, node);


  if (group)
  {
    StringBuffer_appendChar(sb, '(');
  }

  if (numChildren > 0)
  {
    FormulaFormatter_visit( node, ASTNode_getLeftChild(node), sb );
  }

  FormulaFormatter_format(sb, node);

  if (numChildren > 1)
  {
    FormulaFormatter_visit( node, ASTNode_getRightChild(node), sb );
  }

  if (group)
  {
    StringBuffer_appendChar(sb, ')');
  }
}

/** @endcond */

