/**
 * @file    ASTNode.h
 * @brief   Abstract Syntax Tree (AST) for representing formula trees.
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
 * ------------------------------------------------------------------------ -->
 *
 * @class ASTNode
 * @brief Abstract Syntax Tree (AST) representation of a
 * mathematical expression.
 *
 * @htmlinclude not-sbml-warning.html
 *
 * Abstract Syntax Trees (ASTs) are a simple kind of data structure used in
 * libSBML for storing mathematical expressions.  The ASTNode is the
 * cornerstone of libSBML's AST representation.  An AST "node" represents the
 * most basic, indivisible part of a mathematical formula and come in many
 * types.  For instance, there are node types to represent numbers (with
 * subtypes to distinguish integer, real, and rational numbers), names
 * (e.g., constants or variables), simple mathematical operators, logical
 * or relational operators and functions. LibSBML ASTs provide a canonical,
 * in-memory representation for all mathematical formulas regardless of
 * their original format (which might be MathML or might be text strings).
 *
 * An AST @em node in libSBML is a recursive structure containing a pointer
 * to the node's value (which might be, for example, a number or a symbol)
 * and a list of children nodes.  Each ASTNode node may have none, one,
 * two, or more children depending on its type.  The following diagram
 * illustrates an example of how the mathematical expression <code>"1 +
 * 2"</code> is represented as an AST with one @em plus node having two @em
 * integer children nodes for the numbers <code>1</code> and
 * <code>2</code>.  The figure also shows the corresponding MathML
 * representation:
 *
 * @htmlinclude astnode-illustration.html
 *
 * The following are other noteworthy points about the AST representation
 * in libSBML:
 * <ul>
 * <li> A numerical value represented in MathML as a real number with an
 * exponent is preserved as such in the AST node representation, even if
 * the number could be stored in a @c double data type.  This is done
 * so that when an SBML model is read in and then written out again, the
 * amount of change introduced by libSBML to the SBML during the round-trip
 * activity is minimized.
 *  
 * <li> Rational numbers are represented in an AST node using separate
 * numerator and denominator values.  These can be retrieved using the
 * methods ASTNode::getNumerator() and ASTNode::getDenominator().
 * 
 * <li> The children of an ASTNode are other ASTNode objects.  The list of
 * children is empty for nodes that are leaf elements, such as numbers.
 * For nodes that are actually roots of expression subtrees, the list of
 * children points to the parsed objects that make up the rest of the
 * expression.
 * </ul>
 *
 *
 * @if clike <h3><a class="anchor" name="ASTNodeType_t">
 * ASTNodeType_t</a></h3> @else <h3><a class="anchor"
 * name="ASTNodeType_t">The set of possible %ASTNode types</a></h3> @endif@~
 *
 * Every ASTNode has an associated type code to indicate,
 * for example, whether it holds a number or stands for an arithmetic
 * operator.
 * @if clike The type is recorded as a value drawn from the enumeration 
 * @link ASTNode.h::ASTNodeType_t ASTNodeType_t@endlink.@endif
 * @if java The type is recorded as a value drawn from a
 * set of static integer constants defined in the class {@link
 * libsbmlConstants}. Their names begin with the characters @c AST_.@endif
 * @if python The type is recorded as a value drawn from a
 * set of static integer constants defined in the class {@link
 * libsbml}. Their names begin with the characters @c AST_.@endif
 * @if csharp The type is recorded as a value drawn from a
 * set of static integer constants defined in the class {@link
 * libsbml}. Their names begin with the characters @c AST_.@endif
 * The list of possible types is quite long, because it covers all the
 * mathematical functions that are permitted in SBML. The values are shown
 * in the following table:
 *
 * @htmlinclude astnode-types.html
 *
 * The types have the following meanings:
 * <ul>
 * <li> If the node is basic mathematical operator (e.g., @c "+"), then the
 * node's type will be @c AST_PLUS, @c AST_MINUS, @c AST_TIMES, @c AST_DIVIDE,
 * or @c AST_POWER, as appropriate.
 *
 * <li> If the node is a predefined function or operator from %SBML Level&nbsp;1
 * (in the string-based formula syntax used in Level&nbsp;1) or %SBML Levels&nbsp;2 and&nbsp;3
 * (in the subset of MathML used in SBML Levels&nbsp;2 and&nbsp;3), then the node's type
 * will be either <code>AST_FUNCTION_</code><em><span
 * class="placeholder">X</span></em>, <code>AST_LOGICAL_</code><em><span
 * class="placeholder">X</span></em>, or
 * <code>AST_RELATIONAL_</code><em><span class="placeholder">X</span></em>,
 * as appropriate.  (Examples: @c AST_FUNCTION_LOG, @c AST_RELATIONAL_LEQ.)
 *
 * <li> If the node refers to a user-defined function, the node's type will
 * be @c AST_NAME (because it holds the name of the function).
 *
 * <li> If the node is a lambda expression, its type will be @c AST_LAMBDA.
 * 
 * <li> If the node is a predefined constant (@c "ExponentialE", @c "Pi", 
 * @c "True" or @c "False"), then the node's type will be @c AST_CONSTANT_E,
 * @c AST_CONSTANT_PI, @c AST_CONSTANT_TRUE, or @c AST_CONSTANT_FALSE.
 * 
 * <li> (Levels&nbsp;2 and&nbsp;3 only) If the node is the special MathML csymbol @c time,
 * the value of the node will be @c AST_NAME_TIME.  (Note, however, that the
 * MathML csymbol @c delay is translated into a node of type
 * @c AST_FUNCTION_DELAY.  The difference is due to the fact that @c time is a
 * single variable, whereas @c delay is actually a function taking
 * arguments.)
 *
 * <li> (Level&nbsp;3 only) If the node is the special MathML csymbol @c avogadro,
 * the value of the node will be @c AST_NAME_AVOGADRO.
 * 
 * <li> If the node contains a numerical value, its type will be
 * @c AST_INTEGER, @c AST_REAL, @c AST_REAL_E, or @c AST_RATIONAL,
 * as appropriate.
 * </ul>
 *
 * 
 * <h3><a class="anchor" name="math-convert">Converting between ASTs and text strings</a></h3>
 * 
 * The text-string form of mathematical formulas produced by @if clike SBML_formulaToString()@endif@if csharp SBML_formulaToString()@endif@if python libsbml.formulaToString()@endif@if java <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode)">libsbml.formulaToString()</a></code>@endif@~ and
 * read by @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~
 * and
 * @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
 * are in a simple C-inspired infix notation.  A
 * formula in this text-string form can be handed to a program that
 * understands SBML mathematical expressions, or used as part
 * of a translation system.  The libSBML distribution comes with an example
 * program in the @c "examples" subdirectory called @c translateMath that
 * implements an interactive command-line demonstration of translating
 * infix formulas into MathML and vice-versa.
 *
 * The formula strings may contain operators, function calls, symbols, and
 * white space characters.  The allowable white space characters are tab
 * and space.  The following are illustrative examples of formulas
 * expressed in the syntax:
 * 
 * @verbatim
0.10 * k4^2
@endverbatim
 * @verbatim
(vm * s1)/(km + s1)
@endverbatim
 *
 * The following table shows the precedence rules in this syntax.  In the
 * Class column, @em operand implies the construct is an operand, @em
 * prefix implies the operation is applied to the following arguments, @em
 * unary implies there is one argument, and @em binary implies there are
 * two arguments.  The values in the Precedence column show how the order
 * of different types of operation are determined.  For example, the
 * expression <em>a * b + c</em> is evaluated as <em>(a * b) + c</em>
 * because the <code>*</code> operator has higher precedence.  The
 * Associates column shows how the order of similar precedence operations
 * is determined; for example, <em>a - b + c</em> is evaluated as <em>(a -
 * b) + c</em> because the <code>+</code> and <code>-</code> operators are
 * left-associative.  The precedence and associativity rules are taken from
 * the C programming language, except for the symbol <code>^</code>, which
 * is used in C for a different purpose.  (Exponentiation can be invoked
 * using either <code>^</code> or the function @c power.)
 * 
 * @htmlinclude math-precedence-table.html 
 *
 * A program parsing a formula in an SBML model should assume that names
 * appearing in the formula are the identifiers of Species, Parameter,
 * Compartment, FunctionDefinition, Reaction (in SBML Levels&nbsp;2
 * and&nbsp;3), or SpeciesReference (in SBML Level&nbsp;3 only) objects
 * defined in a model.  When a function call is involved, the syntax
 * consists of a function identifier, followed by optional white space,
 * followed by an opening parenthesis, followed by a sequence of zero or
 * more arguments separated by commas (with each comma optionally preceded
 * and/or followed by zero or more white space characters), followed by a
 * closing parenthesis.  There is an almost one-to-one mapping between the
 * list of predefined functions available, and those defined in MathML.
 * All of the MathML functions are recognized; this set is larger than the
 * functions defined in SBML Level&nbsp;1.  In the subset of functions that
 * overlap between MathML and SBML Level&nbsp;1, there exist a few
 * differences.  The following table summarizes the differences between the
 * predefined functions in SBML Level&nbsp;1 and the MathML equivalents in
 * SBML Levels&nbsp;2 and &nbsp;3:
 * 
 * @htmlinclude math-functions.html
 * 
 * @warning @htmlinclude L1-math-syntax-warning.html
 *
 * @if clike @see SBML_parseL3Formula()@endif@~
 * @if csharp @see SBML_parseL3Formula()@endif@~
 * @if python @see libsbml.parseL3Formula()@endif@~
 * @if java @see <code><a href="libsbml.html#parseL3Formula(String formula)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
 * @if clike @see SBML_parseFormula()@endif@~
 * @if csharp @see SBML_parseFormula()@endif@~
 * @if python @see libsbml.parseFormula()@endif@~
 * @if java @see <code><a href="libsbml.html#parseFormula(String formula)">libsbml.parseFormula(String formula)</a></code>@endif@~
 */

#ifndef ASTNode_h
#define ASTNode_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>

#include <sbml/math/FormulaTokenizer.h>
#include <sbml/math/FormulaFormatter.h>
#include <sbml/math/FormulaParser.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/SyntaxChecker.h>

#include <sbml/common/operationReturnValues.h>

LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * @enum  ASTNodeType_t
 * @brief ASTNodeType_t is the enumeration of possible ASTNode types.
 *
 * Each ASTNode has a type whose value is one of the elements of this
 * enumeration.  The types have the following meanings:
 * <ul>
 *
 * <li> If the node is basic mathematical operator (e.g., @c "+"), then the
 * node's type will be @link ASTNodeType_t#AST_PLUS AST_PLUS@endlink, @link
 * ASTNodeType_t#AST_MINUS AST_MINUS@endlink, @link ASTNodeType_t#AST_TIMES
 * AST_TIMES@endlink, @link ASTNodeType_t#AST_DIVIDE AST_DIVIDE@endlink, or
 * @link ASTNodeType_t#AST_POWER AST_POWER@endlink, as appropriate.
 *
 * <li> If the node is a predefined function or operator from %SBML Level&nbsp;1
 * (in the string-based formula syntax used in Level&nbsp;1) or %SBML Level&nbsp;2 and&nbsp;3
 * (in the subset of MathML used in SBML Levels&nbsp;2 and&nbsp;3), then the node's type
 * will be either @c AST_FUNCTION_<em><span
 * class="placeholder">X</span></em>, @c AST_LOGICAL_<em><span
 * class="placeholder">X</span></em>, or @c AST_RELATIONAL_<em><span
 * class="placeholder">X</span></em>, as appropriate.  (Examples: @link
 * ASTNodeType_t#AST_FUNCTION_LOG AST_FUNCTION_LOG@endlink, @link
 * ASTNodeType_t#AST_RELATIONAL_LEQ AST_RELATIONAL_LEQ@endlink.)
 *
 * <li> If the node refers to a user-defined function, the node's type will
 * be @link ASTNodeType_t#AST_NAME AST_NAME@endlink (because it holds the
 * name of the function).
 *
 * <li> If the node is a lambda expression, its type will be @link
 * ASTNodeType_t#AST_LAMBDA AST_LAMBDA@endlink.
 * 
 * <li> If the node is a predefined constant (@c "ExponentialE", @c "Pi",
 * @c "True" or @c "False"), then the node's type will be @link
 * ASTNodeType_t#AST_CONSTANT_E AST_CONSTANT_E@endlink, @link
 * ASTNodeType_t#AST_CONSTANT_PI AST_CONSTANT_PI@endlink, @link
 * ASTNodeType_t#AST_CONSTANT_TRUE AST_CONSTANT_TRUE@endlink, or @link
 * ASTNodeType_t#AST_CONSTANT_FALSE AST_CONSTANT_FALSE@endlink.
 * 
 * <li> (Levels&nbsp;2 and&nbsp;3 only) If the node is the special MathML csymbol @c time,
 * the value of the node will be @link ASTNodeType_t#AST_NAME_TIME
 * AST_NAME_TIME@endlink.  (Note, however, that the MathML csymbol @c delay
 * is translated into a node of type @link ASTNodeType_t#AST_FUNCTION_DELAY
 * AST_FUNCTION_DELAY@endlink.  The difference is due to the fact that @c
 * time is a single variable, whereas @c delay is actually a function
 * taking arguments.)
 *
 * <li> (Level&nbsp;3 only) If the node is the special MathML csymbol @c avogadro,
 * the value of the node will be @c AST_NAME_AVOGADRO.
 *
 * <li> If the node contains a numerical value, its type will be @link
 * ASTNodeType_t#AST_INTEGER AST_INTEGER@endlink, @link
 * ASTNodeType_t#AST_REAL AST_REAL@endlink, @link ASTNodeType_t#AST_REAL_E
 * AST_REAL_E@endlink, or @link ASTNodeType_t#AST_RATIONAL
 * AST_RATIONAL@endlink, as appropriate.  </ul>
 * 
 * @see ASTNode::getType()
 * @see ASTNode::canonicalize()
 */
typedef enum
{
    AST_PLUS    = 43 /* '+' */
  , AST_MINUS   = 45 /* '-' */
  , AST_TIMES   = 42 /* '*' */
  , AST_DIVIDE  = 47 /* '/' */
  , AST_POWER   = 94 /* '^' */  

  , AST_INTEGER = 256
  , AST_REAL
  , AST_REAL_E
  , AST_RATIONAL

  , AST_NAME
  , AST_NAME_AVOGADRO
  , AST_NAME_TIME

  , AST_CONSTANT_E
  , AST_CONSTANT_FALSE
  , AST_CONSTANT_PI
  , AST_CONSTANT_TRUE

  , AST_LAMBDA

  , AST_FUNCTION
  , AST_FUNCTION_ABS
  , AST_FUNCTION_ARCCOS
  , AST_FUNCTION_ARCCOSH
  , AST_FUNCTION_ARCCOT
  , AST_FUNCTION_ARCCOTH
  , AST_FUNCTION_ARCCSC
  , AST_FUNCTION_ARCCSCH
  , AST_FUNCTION_ARCSEC
  , AST_FUNCTION_ARCSECH
  , AST_FUNCTION_ARCSIN
  , AST_FUNCTION_ARCSINH
  , AST_FUNCTION_ARCTAN
  , AST_FUNCTION_ARCTANH
  , AST_FUNCTION_CEILING
  , AST_FUNCTION_COS
  , AST_FUNCTION_COSH
  , AST_FUNCTION_COT
  , AST_FUNCTION_COTH
  , AST_FUNCTION_CSC
  , AST_FUNCTION_CSCH
  , AST_FUNCTION_DELAY
  , AST_FUNCTION_EXP
  , AST_FUNCTION_FACTORIAL
  , AST_FUNCTION_FLOOR
  , AST_FUNCTION_LN
  , AST_FUNCTION_LOG
  , AST_FUNCTION_PIECEWISE
  , AST_FUNCTION_POWER
  , AST_FUNCTION_ROOT
  , AST_FUNCTION_SEC
  , AST_FUNCTION_SECH
  , AST_FUNCTION_SIN
  , AST_FUNCTION_SINH
  , AST_FUNCTION_TAN
  , AST_FUNCTION_TANH

  , AST_LOGICAL_AND
  , AST_LOGICAL_NOT
  , AST_LOGICAL_OR
  , AST_LOGICAL_XOR

  , AST_RELATIONAL_EQ
  , AST_RELATIONAL_GEQ
  , AST_RELATIONAL_GT
  , AST_RELATIONAL_LEQ
  , AST_RELATIONAL_LT
  , AST_RELATIONAL_NEQ

  , AST_UNKNOWN
} ASTNodeType_t;


/**
 * A pointer to a function that takes an ASTNode and returns @c true
 * (non-zero) or @c false (0).
 *
 * @see ASTNode_getListOfNodes()
 * @see ASTNode_fillListOfNodes()
 */
typedef int (*ASTNodePredicate) (const ASTNode_t *node);

LIBSBML_CPP_NAMESPACE_END

#ifdef __cplusplus

LIBSBML_CPP_NAMESPACE_BEGIN

class List;

class ASTNode
{
public:

  /**
   * Creates and returns a new ASTNode.
   *
   * Unless the argument @p type is given, the returned node will by
   * default have a type of @link ASTNodeType_t#AST_UNKNOWN
   * AST_UNKNOWN@endlink.  If the type isn't supplied when caling this
   * constructor, the caller should set the node type to something else as
   * soon as possible using
   * @if clike setType()@else ASTNode::setType(int)@endif.
   *
   * @param type an optional
   * @if clike @link #ASTNodeType_t ASTNodeType_t@endlink@else type@endif@~
   * code indicating the type of node to create.
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  LIBSBML_EXTERN
  ASTNode (ASTNodeType_t type = AST_UNKNOWN);


  /**
   * Creates a new ASTNode from the given Token.  The resulting ASTNode
   * will contain the same data as the Token.
   *
   * @param token the Token to add.
   */
  LIBSBML_EXTERN
  ASTNode (Token_t *token);

  
  /**
   * Copy constructor; creates a deep copy of the given ASTNode.
   *
   * @param orig the ASTNode to be copied.
   */
  LIBSBML_EXTERN
  ASTNode (const ASTNode& orig);
  

  /**
   * Assignment operator for ASTNode.
   */
  LIBSBML_EXTERN
  ASTNode& operator=(const ASTNode& rhs);


  /**
   * Destroys this ASTNode, including any child nodes.
   */
  LIBSBML_EXTERN
  virtual ~ASTNode ();


  /**
   * Frees the name of this ASTNode and sets it to @c NULL.
   * 
   * This operation is only applicable to ASTNode objects corresponding to
   * operators, numbers, or @link ASTNodeType_t#AST_UNKNOWN
   * AST_UNKNOWN@endlink.  This method has no effect on other types of
   * nodes.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   */
  LIBSBML_EXTERN
  int freeName ();


  /**
   * Converts this ASTNode to a canonical form and returns @c true if
   * successful, @c false otherwise.
   *
   * The rules determining the canonical form conversion are as follows:
   * <ul>
   *
   * <li> If the node type is @link ASTNodeType_t#AST_NAME AST_NAME@endlink
   * and the node name matches @c "ExponentialE", @c "Pi", @c "True" or @c
   * "False" the node type is converted to the corresponding 
   * <code>AST_CONSTANT_</code><em><span class="placeholder">X</span></em> type.
   *
   * <li> If the node type is an @link ASTNodeType_t#AST_FUNCTION
   * AST_FUNCTION@endlink and the node name matches an SBML (MathML) function name, logical operator name, or
   * relational operator name, the node is converted to the corresponding
   * <code>AST_FUNCTION_</code><em><span class="placeholder">X</span></em> or
   * <code>AST_LOGICAL_</code><em><span class="placeholder">X</span></em> type.
   *
   * </ul>
   *
   * SBML Level&nbsp;1 function names are searched first; thus, for
   * example, canonicalizing @c log will result in a node type of @link
   * ASTNodeType_t#AST_FUNCTION_LN AST_FUNCTION_LN@endlink.  (See the SBML
   * Level&nbsp;1 Version&nbsp;2 Specification, Appendix C.)
   *
   * Sometimes, canonicalization of a node results in a structural
   * conversion of the node as a result of adding a child.  For example, a
   * node with the SBML Level&nbsp;1 function name @c sqr and a single
   * child node (the argument) will be transformed to a node of type
   * @link ASTNodeType_t#AST_FUNCTION_POWER AST_FUNCTION_POWER@endlink with
   * two children.  The first child will remain unchanged, but the second
   * child will be an ASTNode of type @link ASTNodeType_t#AST_INTEGER
   * AST_INTEGER@endlink and a value of 2.  The function names that result
   * in structural changes are: @c log10, @c sqr, and @c sqrt.
   */
  LIBSBML_EXTERN
  bool canonicalize ();


  /**
   * Adds the given node as a child of this ASTNode.  Child nodes are added
   * in-order, from left to right.
   *
   * @param child the ASTNode instance to add
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note Adding a child to an ASTNode may change the structure of the
   * mathematical formula being represented by the tree structure, and may
   * render the representation invalid.  Callers need to be careful to use
   * this method in the context of other operations to create complete and
   * correct formulas.  The method
   * @if clike isWellFormedASTNode()@else ASTNode::isWellFormedASTNode()@endif@~
   * may also be useful for checking the results of node modifications.
   *
   * @see prependChild(ASTNode* child)
   * @see replaceChild(unsigned int n, ASTNode* child)
   * @see insertChild(unsigned int n, ASTNode* child)
   * @see removeChild(unsigned int n)
   * @see isWellFormedASTNode()
   */
  LIBSBML_EXTERN
  int addChild (ASTNode* child);


  /**
   * Adds the given node as a child of this ASTNode.  This method adds
   * child nodes from right to left.
   *
   * @param child the ASTNode instance to add
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note Prepending a child to an ASTNode may change the structure of the
   * mathematical formula being represented by the tree structure, and may
   * render the representation invalid.
   *
   * @see addChild(ASTNode* child)
   * @see replaceChild(unsigned int n, ASTNode* child)
   * @see insertChild(unsigned int n, ASTNode* child)
   * @see removeChild(unsigned int n)
   */
  LIBSBML_EXTERN
  int prependChild (ASTNode* child);


  /**
   * Removes the nth child of this ASTNode object.
   *
   * @param n unsigned int the index of the child to remove
   *
   * @return integer value indicating success/failure of the
   * function. The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INDEX_EXCEEDS_SIZE LIBSBML_INDEX_EXCEEDS_SIZE @endlink
   *
   * @note Removing a child from an ASTNode may change the structure of the
   * mathematical formula being represented by the tree structure, and may
   * render the representation invalid.
   *
   * @see addChild(ASTNode* child)
   * @see prependChild(ASTNode* child)
   * @see replaceChild(unsigned int n, ASTNode* child)
   * @see insertChild(unsigned int n, ASTNode* child)
   */
  LIBSBML_EXTERN
  int removeChild(unsigned int n);


  /**
   * Replaces the nth child of this ASTNode with the given ASTNode.
   *
   * @param n unsigned int the index of the child to replace
   * @param newChild ASTNode to replace the nth child
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INDEX_EXCEEDS_SIZE LIBSBML_INDEX_EXCEEDS_SIZE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   *
   * @note Replacing a child from an ASTNode may change the structure of the
   * mathematical formula being represented by the tree structure, and may
   * render the representation invalid.
   * 
   * @see addChild(ASTNode* child)
   * @see prependChild(ASTNode* child)
   * @see insertChild(unsigned int n, ASTNode* child)
   * @see removeChild(unsigned int n)
   */
  LIBSBML_EXTERN
  int replaceChild(unsigned int n, ASTNode *newChild);


  /**
   * Insert the given ASTNode at point n in the list of children
   * of this ASTNode.
   *
   * @param n unsigned int the index of the ASTNode being added
   * @param newChild ASTNode to insert as the nth child
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INDEX_EXCEEDS_SIZE LIBSBML_INDEX_EXCEEDS_SIZE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_OBJECT LIBSBML_INVALID_OBJECT @endlink
   *
   * @note Inserting a child into an ASTNode may change the structure of the
   * mathematical formula being represented by the tree structure, and may
   * render the representation invalid.
   * 
   * @see addChild(ASTNode* child)
   * @see prependChild(ASTNode* child)
   * @see replaceChild(unsigned int n, ASTNode* child)
   * @see removeChild(unsigned int n)
   */
  LIBSBML_EXTERN
  int insertChild(unsigned int n, ASTNode *newChild);


  /**
   * Creates a recursive copy of this node and all its children.
   * 
   * @return a copy of this ASTNode and all its children.  The caller owns
   * the returned ASTNode and is reponsible for deleting it.
   */
  LIBSBML_EXTERN
  ASTNode* deepCopy () const;


  /**
   * Get a child of this node according to its index number.
   *
   * @param n the index of the child to get
   * 
   * @return the nth child of this ASTNode or @c NULL if this node has no nth
   * child (<code>n &gt; </code>
   * @if clike getNumChildren()@else ASTNode::getNumChildren()@endif@~
   * <code>- 1</code>).
   */
  LIBSBML_EXTERN
  ASTNode* getChild (unsigned int n) const;


  /**
   * Get the left child of this node.
   * 
   * @return the left child of this ASTNode.  This is equivalent to calling
   * @if clike getChild()@else ASTNode::getChild(unsigned int)@endif@~
   * with an argument of @c 0.
   */
  LIBSBML_EXTERN
  ASTNode* getLeftChild () const;


  /**
   * Get the right child of this node.
   *
   * @return the right child of this ASTNode, or @c NULL if this node has no
   * right child.  If
   * @if clike getNumChildren()@else ASTNode::getNumChildren()@endif@~
   * <code>&gt; 1</code>, then this is equivalent to:
   * @code
   * getChild( getNumChildren() - 1 );
   * @endcode
   */
  LIBSBML_EXTERN
  ASTNode* getRightChild () const;


  /**
   * Get the number of children that this node has.
   * 
   * @return the number of children of this ASTNode, or 0 is this node has
   * no children.
   */
  LIBSBML_EXTERN
  unsigned int getNumChildren () const;


  /**
   * Adds the given XMLNode as a <em>semantic annotation</em> of this ASTNode.
   *
   * @htmlinclude about-semantic-annotations.html
   *
   * @param sAnnotation the annotation to add.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   *
   * @note Although SBML permits the semantic annotation construct in
   * MathML expressions, the truth is that this construct has so far (at
   * this time of this writing, which is early 2011) seen very little use
   * in SBML software.  The full implications of using semantic annotations
   * are still poorly understood.  If you wish to use this construct, we
   * urge you to discuss possible uses and applications on the SBML
   * discussion lists, particularly <a target="_blank"
   * href="http://sbml.org/Forums">sbml-discuss&#64;caltech.edu</a> and/or <a
   * target="_blank"
   * href="http://sbml.org/Forums">sbml-interoperability&#64;caltech.edu</a>.
   */
  LIBSBML_EXTERN
  int addSemanticsAnnotation (XMLNode* sAnnotation);


  /**
   * Get the number of <em>semantic annotation</em> elements inside this node.
   *
   * @htmlinclude about-semantic-annotations.html
   * 
   * @return the number of annotations of this ASTNode.
   *
   * @see ASTNode::addSemanticsAnnotation(XMLNode* sAnnotation)
   */
  LIBSBML_EXTERN
  unsigned int getNumSemanticsAnnotations () const;


  /**
   * Get the nth semantic annotation of this node.
   *
   * @htmlinclude about-semantic-annotations.html
   * 
   * @return the nth annotation of this ASTNode, or @c NULL if this node has
   * no nth annotation (<code>n &gt;</code>
   * @if clike getNumChildren()@else ASTNode::getNumChildren()@endif@~
   * <code>- 1</code>).
   *
   * @see ASTNode::addSemanticsAnnotation(XMLNode* sAnnotation)
   */
  LIBSBML_EXTERN
  XMLNode* getSemanticsAnnotation (unsigned int n) const;


  /**
   * Performs a depth-first search of the tree rooted at this ASTNode
   * object, and returns a List of nodes where the given function
   * <code>predicate(node)</code> returns @c true (non-zero).
   *
   * For portability between different programming languages, the predicate
   * is passed in as a pointer to a function.  @if clike The function
   * definition must have the type @link ASTNode::ASTNodePredicate
   * ASTNodePredicate @endlink, which is defined as
   * @code
   * int (*ASTNodePredicate) (const ASTNode_t *node);
   * @endcode
   * where a return value of non-zero represents @c true and zero
   * represents @c false. @endif
   *
   * @param predicate the predicate to use
   *
   * @return the list of nodes for which the predicate returned @c true
   * (non-zero).  The List returned is owned by the caller and should be
   * deleted after the caller is done using it.  The ASTNode objects in the
   * list; however, are not owned by the caller (as they still belong to
   * the tree itself), and therefore should not be deleted.
   */
  LIBSBML_EXTERN
  List* getListOfNodes (ASTNodePredicate predicate) const;


  /**
   * Performs a depth-first search of the tree rooted at this ASTNode
   * object, and adds to the list @p lst the nodes where the given function
   * <code>predicate(node)</code> returns @c true (non-zero).
   *
   * This method is identical to getListOfNodes(ASTNodePredicate predicate) const, 
   * except that instead of creating a new List object, it uses the one passed
   * in as argument @p lst. 
   *
   * For portability between different programming languages, the predicate
   * is passed in as a pointer to a function.  The function definition must
   * have the type @link ASTNode.h::ASTNodePredicate ASTNodePredicate
   * @endlink, which is defined as
   * @code
   * int (*ASTNodePredicate) (const ASTNode_t *node);
   * @endcode
   * where a return value of non-zero represents @c true and zero
   * represents @c false.
   *
   * @param predicate the predicate to use.
   *
   * @param lst the List to which ASTNode objects should be added.
   *
   * @see getListOfNodes(ASTNodePredicate predicate) const
   */
  LIBSBML_EXTERN
  void fillListOfNodes (ASTNodePredicate predicate, List* lst) const;


  /**
   * Get the value of this node as a single character.  This function
   * should be called only when
   * @if clike getType()@else ASTNode::getType()@endif@~ returns
   * @link ASTNodeType_t#AST_PLUS AST_PLUS@endlink,
   * @link ASTNodeType_t#AST_MINUS AST_MINUS@endlink,
   * @link ASTNodeType_t#AST_TIMES AST_TIMES@endlink,
   * @link ASTNodeType_t#AST_DIVIDE AST_DIVIDE@endlink or
   * @link ASTNodeType_t#AST_POWER AST_POWER@endlink.
   * 
   * @return the value of this ASTNode as a single character
   */
  LIBSBML_EXTERN
  char getCharacter () const;


  /**
   * Get the id of this ASTNode.  
   * 
   * @return the mathml id of this ASTNode.
   */
  LIBSBML_EXTERN
  std::string getId () const;


  /**
   * Get the class of this ASTNode.  
   * 
   * @return the mathml class of this ASTNode.
   */
  LIBSBML_EXTERN
  std::string getClass () const;


  /**
   * Get the style of this ASTNode.  
   * 
   * @return the mathml style of this ASTNode.
   */
  LIBSBML_EXTERN
  std::string getStyle () const;


  /**
   * Get the value of this node as an integer. This function should be
   * called only when
   * @if clike getType()@else ASTNode::getType()@endif@~
   * <code>== @link ASTNodeType_t#AST_INTEGER AST_INTEGER@endlink</code>.
   * 
   * @return the value of this ASTNode as a (<code>long</code>) integer. 
   */
  LIBSBML_EXTERN
  long getInteger () const;


  /**
   * Get the value of this node as a string.  This function may be called
   * on nodes that (1) are not operators, i.e., nodes for which
   * @if clike isOperator()@else ASTNode::isOperator()@endif@~
   * returns @c false, and (2) are not numbers, i.e.,
   * @if clike isNumber()@else ASTNode::isNumber()@endif@~ returns @c false.
   * 
   * @return the value of this ASTNode as a string.
   */
  LIBSBML_EXTERN
  const char* getName () const;


  /**
   * Get the value of this operator node as a string.  This function may be called
   * on nodes that are operators, i.e., nodes for which
   * @if clike isOperator()@else ASTNode::isOperator()@endif@~
   * returns @c true.
   * 
   * @return the name of this operator ASTNode as a string (or NULL if not an operator).
   */
  LIBSBML_EXTERN
  const char* getOperatorName () const;


  /**
   * Get the value of the numerator of this node.  This function should be
   * called only when
   * @if clike getType()@else ASTNode::getType()@endif@~
   * <code>== @link ASTNodeType_t#AST_RATIONAL AST_RATIONAL@endlink</code>.
   * 
   * @return the value of the numerator of this ASTNode.  
   */
  LIBSBML_EXTERN
  long getNumerator () const;


  /**
   * Get the value of the denominator of this node.  This function should
   * be called only when
   * @if clike getType()@else ASTNode::getType()@endif@~
   * <code>== @link ASTNodeType_t#AST_RATIONAL AST_RATIONAL@endlink</code>.
   * 
   * @return the value of the denominator of this ASTNode.
   */
  LIBSBML_EXTERN
  long getDenominator () const;


  /**
   * Get the real-numbered value of this node.  This function
   * should be called only when
   * @if clike isReal()@else ASTNode::isReal()@endif@~
   * <code>== true</code>.
   *
   * This function performs the necessary arithmetic if the node type is
   * @link ASTNodeType_t#AST_REAL_E AST_REAL_E@endlink (<em>mantissa *
   * 10<sup> exponent</sup></em>) or @link ASTNodeType_t#AST_RATIONAL
   * AST_RATIONAL@endlink (<em>numerator / denominator</em>).
   * 
   * @return the value of this ASTNode as a real (double).
   */
  LIBSBML_EXTERN
  double getReal () const;


  /**
   * Get the mantissa value of this node.  This function should be called
   * only when @if clike getType()@else ASTNode::getType()@endif@~
   * returns @link ASTNodeType_t#AST_REAL_E AST_REAL_E@endlink
   * or @link ASTNodeType_t#AST_REAL AST_REAL@endlink.
   * If @if clike getType()@else ASTNode::getType()@endif@~
   * returns @link ASTNodeType_t#AST_REAL AST_REAL@endlink,
   * this method is identical to
   * @if clike getReal()@else ASTNode::getReal()@endif.
   * 
   * @return the value of the mantissa of this ASTNode. 
   */
  LIBSBML_EXTERN
  double getMantissa () const;


  /**
   * Get the exponent value of this ASTNode.  This function should be
   * called only when
   * @if clike getType()@else ASTNode::getType()@endif@~
   * returns @link ASTNodeType_t#AST_REAL_E AST_REAL_E@endlink
   * or @link ASTNodeType_t#AST_REAL AST_REAL@endlink.
   * 
   * @return the value of the exponent of this ASTNode.
   */
  LIBSBML_EXTERN
  long getExponent () const;


  /**
   * Get the precedence of this node in the infix math syntax of SBML
   * Level&nbsp;1.  For more information about the infix syntax, see the
   * discussion about <a href="#math-convert">text string formulas</a> at
   * the top of the documentation for ASTNode.
   * 
   * @return an integer indicating the precedence of this ASTNode
   */
  LIBSBML_EXTERN
  int getPrecedence () const;


  /**
   * Get the type of this ASTNode.  The value returned is one of the
   * enumeration values such as @link ASTNodeType_t#AST_LAMBDA
   * AST_LAMBDA@endlink, @link ASTNodeType_t#AST_PLUS AST_PLUS@endlink,
   * etc.
   * 
   * @return the type of this ASTNode.
   */
  LIBSBML_EXTERN
  ASTNodeType_t getType () const;


  /**
   * Get the units of this ASTNode.  
   *
   * @htmlinclude about-sbml-units-attrib.html
   * 
   * @return the units of this ASTNode.
   *
   * @note The <code>sbml:units</code> attribute is only available in SBML
   * Level&nbsp;3.  It may not be used in Levels 1&ndash;2 of SBML.
   * 
   * @if clike @see SBML_parseL3Formula()@endif@~
   * @if csharp @see SBML_parseL3Formula()@endif@~
   * @if python @see libsbml.parseL3Formula()@endif@~
   * @if java @see <code><a href="libsbml.html#parseL3Formula(String formula)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
   */
  LIBSBML_EXTERN
  std::string getUnits () const;


  /**
   * Predicate returning @c true (non-zero) if this node is the special 
   * symbol @c avogadro.  The predicate returns @c false (zero) otherwise.
   * 
   * @return @c true if this ASTNode is the special symbol avogadro.
   *
   * @if clike @see SBML_parseL3Formula()@endif@~
   * @if csharp @see SBML_parseL3Formula()@endif@~
   * @if python @see libsbml.parseL3Formula()@endif@~
   * @if java @see <code><a href="libsbml.html#parseL3Formula(String formula)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
   */
  LIBSBML_EXTERN
  bool isAvogadro () const;


  /**
   * Predicate returning @c true (non-zero) if this node has a boolean type
   * (a logical operator, a relational operator, or the constants @c true
   * or @c false).
   *
   * @return true if this ASTNode is a boolean, false otherwise.
   */
  LIBSBML_EXTERN
  bool isBoolean () const;


  /**
   * Predicate returning @c true (non-zero) if this node returns a boolean type
   * or @c false (zero) otherwise.
   *
   * This function looks at the whole ASTNode rather than just the top 
   * level of the ASTNode. Thus it will consider return values from
   * piecewise statements.  In addition, if this ASTNode uses a function
   * call, the return value of the functionDefinition will be determined.
   * Note that this is only possible where the ASTNode can trace its parent
   * Model, that is, the ASTNode must represent the math element of some
   * SBML object that has already been added to an instance of an SBMLDocument.
   *
   * @see isBoolean()
   *
   * @return true if this ASTNode returns a boolean, false otherwise.
   */
  LIBSBML_EXTERN
  bool returnsBoolean (const Model* model=NULL) const;


  /**
   * Predicate returning @c true (non-zero) if this node represents a MathML
   * constant (e.g., @c true, @c Pi).
   * 
   * @return @c true if this ASTNode is a MathML constant, @c false otherwise.
   * 
   * @note this function will also return @c true for @link
   * ASTNodeType_t#AST_NAME_AVOGADRO AST_NAME_AVOGADRO@endlink in SBML Level&nbsp;3.
   */
  LIBSBML_EXTERN
  bool isConstant () const;


  /**
   * Predicate returning @c true (non-zero) if this node represents a
   * MathML function (e.g., <code>abs()</code>), or an SBML Level&nbsp;1
   * function, or a user-defined function.
   * 
   * @return @c true if this ASTNode is a function, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isFunction () const;


  /**
   * Predicate returning @c true (non-zero) if this node represents
   * the special IEEE 754 value infinity, @c false (zero) otherwise.
   *
   * @return @c true if this ASTNode is the special IEEE 754 value infinity,
   * @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isInfinity () const;


  /**
   * Predicate returning @c true (non-zero) if this node contains an
   * integer value, @c false (zero) otherwise.
   *
   * @return @c true if this ASTNode is of type @link
   * ASTNodeType_t#AST_INTEGER AST_INTEGER@endlink, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isInteger () const;


  /**
   * Predicate returning @c true (non-zero) if this node is a MathML
   * <code>&lt;lambda&gt;</code>, @c false (zero) otherwise.
   * 
   * @return @c true if this ASTNode is of type @link ASTNodeType_t#AST_LAMBDA
   * AST_LAMBDA@endlink, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isLambda () const;


  /**
   * Predicate returning @c true (non-zero) if this node represents a 
   * @c log10 function, @c false (zero) otherwise.  More precisely, this
   * predicate returns @c true if the node type is @link
   * ASTNodeType_t#AST_FUNCTION_LOG AST_FUNCTION_LOG@endlink with two
   * children, the first of which is an @link ASTNodeType_t#AST_INTEGER
   * AST_INTEGER@endlink equal to 10.
   * 
   * @return @c true if the given ASTNode represents a log10() function, @c
   * false otherwise.
   *
   * @if clike @see SBML_parseL3Formula()@endif@~
   * @if csharp @see SBML_parseL3Formula()@endif@~
   * @if python @see libsbml.parseL3Formula()@endif@~
   * @if java @see <code><a href="libsbml.html#parseL3Formula(String formula)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
   */
  LIBSBML_EXTERN
  bool isLog10 () const;


  /**
   * Predicate returning @c true (non-zero) if this node is a MathML
   * logical operator (i.e., @c and, @c or, @c not, @c xor).
   * 
   * @return @c true if this ASTNode is a MathML logical operator
   */
  LIBSBML_EXTERN
  bool isLogical () const;


  /**
   * Predicate returning @c true (non-zero) if this node is a user-defined
   * variable name in SBML L1, L2 (MathML), or the special symbols @c delay
   * or @c time.  The predicate returns @c false (zero) otherwise.
   * 
   * @return @c true if this ASTNode is a user-defined variable name in SBML
   * L1, L2 (MathML) or the special symbols delay or time.
   */
  LIBSBML_EXTERN
  bool isName () const;


  /**
   * Predicate returning @c true (non-zero) if this node represents the
   * special IEEE 754 value "not a number" (NaN), @c false (zero)
   * otherwise.
   * 
   * @return @c true if this ASTNode is the special IEEE 754 NaN.
   */
  LIBSBML_EXTERN
  bool isNaN () const;


  /**
   * Predicate returning @c true (non-zero) if this node represents the
   * special IEEE 754 value "negative infinity", @c false (zero) otherwise.
   * 
   * @return @c true if this ASTNode is the special IEEE 754 value negative
   * infinity, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isNegInfinity () const;


  /**
   * Predicate returning @c true (non-zero) if this node contains a number,
   * @c false (zero) otherwise.  This is functionally equivalent to the
   * following code:
   * @code
   *   isInteger() || isReal()
   * @endcode
   * 
   * @return @c true if this ASTNode is a number, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isNumber () const;


  /**
   * Predicate returning @c true (non-zero) if this node is a mathematical
   * operator, meaning, <code>+</code>, <code>-</code>, <code>*</code>, 
   * <code>/</code> or <code>^</code> (power).
   * 
   * @return @c true if this ASTNode is an operator.
   */
  LIBSBML_EXTERN
  bool isOperator () const;


  /**
   * Predicate returning @c true (non-zero) if this node is the MathML
   * <code>&lt;piecewise&gt;</code> construct, @c false (zero) otherwise.
   * 
   * @return @c true if this ASTNode is a MathML @c piecewise function
   */
  LIBSBML_EXTERN
  bool isPiecewise () const;


  /**
   * Predicate returning @c true (non-zero) if this node represents a rational
   * number, @c false (zero) otherwise.
   * 
   * @return @c true if this ASTNode is of type @link
   * ASTNodeType_t#AST_RATIONAL AST_RATIONAL@endlink.
   */
  LIBSBML_EXTERN
  bool isRational () const;


  /**
   * Predicate returning @c true (non-zero) if this node can represent a
   * real number, @c false (zero) otherwise.  More precisely, this node
   * must be of one of the following types: @link ASTNodeType_t#AST_REAL
   * AST_REAL@endlink, @link ASTNodeType_t#AST_REAL_E AST_REAL_E@endlink or
   * @link ASTNodeType_t#AST_RATIONAL AST_RATIONAL@endlink.
   * 
   * @return @c true if the value of this ASTNode can represented as a real
   * number, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isReal () const;


  /**
   * Predicate returning @c true (non-zero) if this node is a MathML
   * relational operator, meaning <code>==</code>, <code>&gt;=</code>, 
   * <code>&gt;</code>, <code>&lt;</code>, and <code>!=</code>.
   * 
   * @return @c true if this ASTNode is a MathML relational operator, @c
   * false otherwise
   */
  LIBSBML_EXTERN
  bool isRelational () const;


  /**
   * Predicate returning @c true (non-zero) if this node represents a
   * square root function, @c false (zero) otherwise.  More precisely, the
   * node type must be @link ASTNodeType_t#AST_FUNCTION_ROOT
   * AST_FUNCTION_ROOT@endlink with two children, the first of which is an
   * @link ASTNodeType_t#AST_INTEGER AST_INTEGER@endlink node having value
   * equal to 2.
   * 
   * @return @c true if the given ASTNode represents a sqrt() function,
   * @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isSqrt () const;


  /**
   * Predicate returning @c true (non-zero) if this node is a unary minus
   * operator, @c false (zero) otherwise.  A node is defined as a unary
   * minus node if it is of type @link ASTNodeType_t#AST_MINUS
   * AST_MINUS@endlink and has exactly one child.
   * 
   * For numbers, unary minus nodes can be "collapsed" by negating the
   * number.  In fact, 
   * @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~
   * does this during its parsing process, and 
   * @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
   * has a configuration option that allows this behavior to be turned
   * on or off.  However, unary minus nodes for symbols
   * (@link ASTNodeType_t#AST_NAME AST_NAME@endlink) cannot
   * be "collapsed", so this predicate function is necessary.
   * 
   * @return @c true if this ASTNode is a unary minus, @c false otherwise.
   *
   * @if clike @see SBML_parseL3Formula()@endif@~
   * @if csharp @see SBML_parseL3Formula()@endif@~
   * @if python @see libsbml.parseL3Formula()@endif@~
   * @if java @see <code><a href="libsbml.html#parseL3Formula(String formula)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
   */
  LIBSBML_EXTERN
  bool isUMinus () const;


  /**
   * Predicate returning @c true (non-zero) if this node is a unary plus
   * operator, @c false (zero) otherwise.  A node is defined as a unary
   * minus node if it is of type @link ASTNodeType_t#AST_MINUS
   * AST_MINUS@endlink and has exactly one child.
   *
   * @return @c true if this ASTNode is a unary plus, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isUPlus () const;


  /**
   * Predicate returning @c true (non-zero) if this node has an unknown type.
   * 
   * "Unknown" nodes have the type @link ASTNodeType_t#AST_UNKNOWN
   * AST_UNKNOWN@endlink.  Nodes with unknown types will not appear in an
   * ASTNode tree returned by libSBML based upon valid SBML input; the only
   * situation in which a node with type @link ASTNodeType_t#AST_UNKNOWN
   * AST_UNKNOWN@endlink may appear is immediately after having create a
   * new, untyped node using the ASTNode constructor.  Callers creating
   * nodes should endeavor to set the type to a valid node type as soon as
   * possible after creating new nodes.
   * 
   * @return @c true if this ASTNode is of type @link
   * ASTNodeType_t#AST_UNKNOWN AST_UNKNOWN@endlink, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool isUnknown () const;


  /**
   * Predicate returning @c true (non-zero) if this node has the mathml attribute
   * <code>id</code>.
   * 
   * @return true if this ASTNode has an attribute id, false otherwise.
   */
  LIBSBML_EXTERN
  bool isSetId() const;

  
  /**
   * Predicate returning @c true (non-zero) if this node has the mathml attribute
   * <code>class</code>.
   * 
   * @return true if this ASTNode has an attribute class, false otherwise.
   */
  LIBSBML_EXTERN
  bool isSetClass() const;


  /**
   * Predicate returning @c true (non-zero) if this node has the mathml attribute
   * <code>style</code>.
   * 
   * @return true if this ASTNode has an attribute style, false otherwise.
   */
  LIBSBML_EXTERN
  bool isSetStyle() const;

    
  /**
   * Predicate returning @c true (non-zero) if this node has the attribute
   * <code>sbml:units</code>.
   *
   * @htmlinclude about-sbml-units-attrib.html
   * 
   * @return @c true if this ASTNode has units associated with it, @c false otherwise.
   *
   * @note The <code>sbml:units</code> attribute is only available in SBML
   * Level&nbsp;3.  It may not be used in Levels 1&ndash;2 of SBML.
   */
  LIBSBML_EXTERN
  bool isSetUnits() const;
  
  
  /**
   * Predicate returning @c true (non-zero) if this node or any of its
   * children nodes have the attribute <code>sbml:units</code>.
   *
   * @htmlinclude about-sbml-units-attrib.html
   * 
   * @return @c true if this ASTNode or its children has units associated
   * with it, @c false otherwise.
   *
   * @note The <code>sbml:units</code> attribute is only available in SBML
   * Level&nbsp;3.  It may not be used in Levels 1&ndash;2 of SBML.
   */
  LIBSBML_EXTERN
  bool hasUnits() const;
  
  
  /**
   * Sets the value of this ASTNode to the given character.  If character
   * is one of @c +, @c -, <code>*</code>, <code>/</code> or @c ^, the node
   * type will be set accordingly.  For all other characters, the node type
   * will be set to @link ASTNodeType_t#AST_UNKNOWN AST_UNKNOWN@endlink.
   *
   * @param value the character value to which the node's value should be
   * set.
   *
   * @return integer value indicating success/failure of the function.  The
   * possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setCharacter (char value);

  /**
   * Sets the mathml id of this ASTNode to id.
   *
   * @param id @c string representing the identifier.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setId (std::string id);

  /**
   * Sets the mathml class of this ASTNode to className.
   *
   * @param className @c string representing the mathml class for this node.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setClass (std::string className);

  /**
   * Sets the mathml style of this ASTNode to style.
   *
   * @param style @c string representing the identifier.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setStyle (std::string style);

  /**
   * Sets the value of this ASTNode to the given name.
   *
   * As a side-effect, this ASTNode object's type will be reset to
   * @link ASTNodeType_t#AST_NAME AST_NAME@endlink if (and <em>only
   * if</em>) the ASTNode was previously an operator (
   * @if clike isOperator()@else ASTNode::isOperator()@endif@~
   * <code>== true</code>), number (
   * @if clike isNumber()@else ASTNode::isNumber()@endif@~
   * <code>== true</code>), or unknown.
   * This allows names to be set for @link ASTNodeType_t#AST_FUNCTION
   * AST_FUNCTION@endlink nodes and the like.
   *
   * @param name the string containing the name to which this node's value
   * should be set
   *
   * @return integer value indicating success/failure of the function.  The
   * possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setName (const char *name);


  /**
   * Sets the value of this ASTNode to the given integer and sets the node
   * type to @link ASTNodeType_t#AST_INTEGER AST_INTEGER@endlink.
   *
   * @param value the integer to which this node's value should be set
   *
   * @return integer value indicating success/failure of the function.  The
   * possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setValue (int value);


  /**
   * Sets the value of this ASTNode to the given (@c long) integer and sets
   * the node type to @link ASTNodeType_t#AST_INTEGER AST_INTEGER@endlink.
   *
   * @param value the integer to which this node's value should be set
   *
   * @return integer value indicating success/failure of the function.  The
   * possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setValue (long value);


  /**
   * Sets the value of this ASTNode to the given rational in two parts: the
   * numerator and denominator.  The node type is set to @link
   * ASTNodeType_t#AST_RATIONAL AST_RATIONAL@endlink.
   *
   * @param numerator the numerator value of the rational
   * @param denominator the denominator value of the rational
   *
   * @return integer value indicating success/failure of the function.  The
   * possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setValue (long numerator, long denominator);


  /**
   * Sets the value of this ASTNode to the given real (@c double) and sets
   * the node type to @link ASTNodeType_t#AST_REAL AST_REAL@endlink.
   *
   * This is functionally equivalent to:
   * @code
   * setValue(value, 0);
   * @endcode
   *
   * @param value the @c double format number to which this node's value
   * should be set
   *
   * @return integer value indicating success/failure of the function.  The
   * possible values returned by this function are: @li @link
   * OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS
   * LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setValue (double value);


  /**
   * Sets the value of this ASTNode to the given real (@c double) in two
   * parts: the mantissa and the exponent.  The node type is set to
   * @link ASTNodeType_t#AST_REAL_E AST_REAL_E@endlink.
   *
   * @param mantissa the mantissa of this node's real-numbered value
   * @param exponent the exponent of this node's real-numbered value
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setValue (double mantissa, long exponent);


  /**
   * Sets the type of this ASTNode to the given type code.  A side-effect
   * of doing this is that any numerical values previously stored in this
   * node are reset to zero.
   *
   * @param type the type to which this node should be set
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   */
  LIBSBML_EXTERN
  int setType (ASTNodeType_t type);


  /**
   * Sets the units of this ASTNode to units.
   *
   * The units will be set @em only if this ASTNode object represents a
   * MathML <code>&lt;cn&gt;</code> element, i.e., represents a number.
   * Callers may use
   * @if clike isNumber()@else ASTNode::isNumber()@endif@~
   * to inquire whether the node is of that type.
   *
   * @htmlinclude about-sbml-units-attrib.html
   *
   * @param units @c string representing the unit identifier.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_ATTRIBUTE_VALUE LIBSBML_INVALID_ATTRIBUTE_VALUE @endlink
   *
   * @note The <code>sbml:units</code> attribute is only available in SBML
   * Level&nbsp;3.  It may not be used in Levels 1&ndash;2 of SBML.
   */
  LIBSBML_EXTERN
  int setUnits (std::string units);


  /**
   * Swap the children of this ASTNode object with the children of the
   * given ASTNode object.
   *
   * @param that the other node whose children should be used to replace
   * <em>this</em> node's children
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  LIBSBML_EXTERN
  int swapChildren (ASTNode *that);


  /**
   * Renames all the SIdRef attributes on this node and any child node
   */
  LIBSBML_EXTERN
  virtual void renameSIdRefs(const std::string& oldid, const std::string& newid);


  /**
   * Renames all the UnitSIdRef attributes on this node and any child node.
   * (The only place UnitSIDRefs appear in MathML <code>&lt;cn&gt;</code> elements.)
   */
  LIBSBML_EXTERN
  virtual void renameUnitSIdRefs(const std::string& oldid, const std::string& newid);


  /** @cond doxygen-libsbml-internal */
  /**
   * Replace any nodes of type AST_NAME with the name 'id' from the child 'math' object with the provided ASTNode. 
   *
   */
  LIBSBML_EXTERN
  virtual void replaceIDWithFunction(const std::string& id, const ASTNode* function);
  /** @endcond */


  /** @cond doxygen-libsbml-internal */
  /**
   * Replaces any 'AST_NAME_TIME' nodes with a node that multiplies time by the given function.
   *
   */
  LIBSBML_EXTERN
  virtual void multiplyTimeBy(const ASTNode* function);
  /** @endcond */


  /**
   * Unsets the units of this ASTNode.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_UNEXPECTED_ATTRIBUTE LIBSBML_UNEXPECTED_ATTRIBUTE @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  LIBSBML_EXTERN
  int unsetUnits ();

  /**
   * Unsets the mathml id of this ASTNode.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  LIBSBML_EXTERN
  int unsetId ();


  /**
   * Unsets the mathml class of this ASTNode.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  LIBSBML_EXTERN
  int unsetClass ();


  /**
   * Unsets the mathml style of this ASTNode.
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  LIBSBML_EXTERN
  int unsetStyle ();


  /** @cond doxygen-libsbml-internal */

  /**
   * Sets the flag indicating that this ASTNode has semantics attached.
   *
   * @htmlinclude about-semantic-annotations.html
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setSemanticsFlag();


  /**
   * Unsets the flag indicating that this ASTNode has semantics attached.
   *
   * @htmlinclude about-semantic-annotations.html
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int unsetSemanticsFlag();


  /**
   * Gets the flag indicating that this ASTNode has semantics attached.
   *
   * @htmlinclude about-semantic-annotations.html
   *
   * @return @c true if this node has semantics attached, @c false otherwise.
   */
  LIBSBML_EXTERN
  bool getSemanticsFlag() const;


  /**
   * Sets the attribute "definitionURL".
   *
   * @return integer value indicating success/failure of the
   * function.  The possible values returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  LIBSBML_EXTERN
  int setDefinitionURL(XMLAttributes url);

  /** @endcond */


  /**
   * Gets the MathML @c definitionURL attribute value.
   *
   * @return the value of the @c definitionURL attribute, in the form of
   * a libSBML XMLAttributes object.
   */
  LIBSBML_EXTERN
  XMLAttributes* getDefinitionURL() const;


  /**
   * Replaces occurences of a given name within this ASTNode with the
   * name/value/formula represented by @p arg.
   * 
   * For example, if the formula in this ASTNode is <code>x + y</code>,
   * then the <code>&lt;bvar&gt;</code> is @c x and @c arg is an ASTNode
   * representing the real value @c 3.  This method substitutes @c 3 for @c
   * x within this ASTNode object.
   *
   * @param bvar a string representing the variable name to be substituted
   * @param arg an ASTNode representing the name/value/formula to substitute
   */
  LIBSBML_EXTERN
  void replaceArgument(const std::string bvar, ASTNode * arg);


  /** @cond doxygen-libsbml-internal */

  /**
   * Sets the parent SBML object.
   * 
   * @param sb the parent SBML object of this ASTNode.
   */
  LIBSBML_EXTERN
  void setParentSBMLObject(SBase * sb);

  /** @endcond */


  /**
   * Returns the parent SBML object.
   * 
   * @return the parent SBML object of this ASTNode.
   */
  LIBSBML_EXTERN
  SBase * getParentSBMLObject() const;


  /**
   * Reduces this ASTNode to a binary tree.
   * 
   * Example: if this ASTNode is <code>and(x, y, z)</code>, then the 
   * formula of the reduced node is <code>and(and(x, y), z)</code>.  The
   * operation replaces the formula stored in the current ASTNode object.
   */
  LIBSBML_EXTERN
  void reduceToBinary();

  
 /**
  * Sets the user data of this node. This can be used by the application
  * developer to attach custom information to the node. In case of a deep
  * copy this attribute will passed as it is. The attribute will be never
  * interpreted by this class.
  * 
  * @param userData specifies the new user data. 
  *
  * @return integer value indicating success/failure of the
  * function.  The possible values returned by this function are:
  * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
  * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
  */
  LIBSBML_EXTERN
  int setUserData(void *userData);


 /**
  * Returns the user data that has been previously set via setUserData().
  *
  * @return the user data of this node, or @c NULL if no user data has been set.
  *
  * @if clike
  * @see ASTNode::setUserData
  * @endif@~
  */
  LIBSBML_EXTERN
  void *getUserData() const;


 /**
  * Predicate returning @c true or @c false depending on whether this
  * ASTNode is well-formed.
  *
  * @note An ASTNode may be well-formed, with each node and its children
  * having the appropriate number of children for the given type, but may
  * still be invalid in the context of its use within an SBML model.
  *
  * @return @c true if this ASTNode is well-formed, @c false otherwise.
  *
  * @see hasCorrectNumberArguments()
  */
  LIBSBML_EXTERN
  bool isWellFormedASTNode() const;


 /**
  * Predicate returning @c true or @c false depending on whether this
  * ASTNode has the correct number of children for it's type.
  *
  * For example, an ASTNode with type @link ASTNodeType_t#AST_PLUS
  * AST_PLUS@endlink expects 2 child nodes.
  *
  * @note This function performs a check on the toplevel node only.
  * Child nodes are not checked.
  *
  * @return @c true if this ASTNode is has appropriate number of children
  * for it's type, @c false otherwise.
  *
  * @see isWellFormedASTNode()
  */
  LIBSBML_EXTERN
  bool hasCorrectNumberArguments() const;

  /** @cond doxygen-libsbml-internal */
    
  bool isBvar() const { return mIsBvar; };
  void setBvar() { mIsBvar = true; };

  /** @endcond */

protected:
  /** @cond doxygen-libsbml-internal */

  /**
   * Internal helper function for canonicalize().
   */

  bool canonicalizeConstant   ();
  bool canonicalizeFunction   ();
  bool canonicalizeFunctionL1 ();
  bool canonicalizeLogical    ();
  bool canonicalizeRelational ();


  ASTNodeType_t mType;

  char   mChar;
  char*  mName;
  long   mInteger;
  double mReal;
  long mDenominator;
  long mExponent;

  XMLAttributes* mDefinitionURL;
  bool hasSemantics;

  List *mChildren;

  List *mSemanticsAnnotations;

  SBase *mParentSBMLObject;

  std::string mUnits;

  // additional MathML attributes
  std::string mId;
  std::string mClass;
  std::string mStyle;

  bool mIsBvar;
  void *mUserData;
  
  friend class MathMLFormatter;
  friend class MathMLHandler;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


/**
 * Creates a new ASTNode and returns a pointer to it.  The returned node
 * will have a type of @c AST_UNKNOWN and should be set to something else as
 * soon as possible.
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_create (void);

/**
 * Creates a new ASTNode and sets its type to the given ASTNodeType.
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_createWithType (ASTNodeType_t type);

/**
 * Creates a new ASTNode from the given Token and returns a pointer to it.
 * The returned ASTNode will contain the same data as the Token.
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_createFromToken (Token_t *token);


/**
 * Frees the given ASTNode including any child nodes.
 */
LIBSBML_EXTERN
void
ASTNode_free (ASTNode_t *node);

/**
 * Frees the name of this ASTNode and sets it to @c NULL.
 * 
 * This operation is only applicable to ASTNode objects corresponding to
 * operators, numbers, or @c AST_UNKNOWN.  This method will have no
 * effect on other types of nodes.
 */
LIBSBML_EXTERN
int
ASTNode_freeName (ASTNode_t *node);


/**
 * Attempts to convert this ASTNode to a canonical form and returns @c true
 * (non-zero) if the conversion succeeded, @c false (0) otherwise.
 *
 * The rules determining the canonical form conversion are as follows:
 *
 *   1. If the node type is @c AST_NAME and the node name matches
 *   "ExponentialE", "Pi", "True" or "False" the node type is converted to
 *   the corresponding @c AST_CONSTANT type.
 *
 *   2. If the node type is an AST_FUNCTION and the node name matches an L1
 *   or L2 (MathML) function name, logical operator name, or relational
 *   operator name, the node is converted to the corresponding @c AST_FUNCTION,
 *   @c AST_LOGICAL or @c AST_CONSTANT type.
 *
 * L1 function names are searched first, so canonicalizing "log" will
 * result in a node type of @c AST_FUNCTION_LN (see L1 Specification,
 * Appendix C).
 *
 * Some canonicalizations result in a structural converion of the nodes (by
 * adding a child).  For example, a node with L1 function name "sqr" and a
 * single child node (the argument) will be transformed to a node of type
 * @c AST_FUNCTION_POWER with two children.  The first child will remain
 * unchanged, but the second child will be an ASTNode of type @c AST_INTEGER
 * and a value of 2.  The function names that result in structural changes
 * are: log10, sqr and sqrt.
 */
LIBSBML_EXTERN
int
ASTNode_canonicalize (ASTNode_t *node);


/**
 * Adds the given node as a child of this ASTNode.  Child nodes are added
 * in-order from "left-to-right".
 */
LIBSBML_EXTERN
int
ASTNode_addChild (ASTNode_t *node, ASTNode_t *child);

/**
 * Adds the given node as a child of this ASTNode.  This method adds child
 * nodes from "right-to-left".
 */
LIBSBML_EXTERN
int
ASTNode_prependChild (ASTNode_t *node, ASTNode_t *child);

/**
 * @return a copy of this ASTNode and all its children.  The caller owns
 * the returned ASTNode and is reponsible for freeing it.
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_deepCopy (const ASTNode_t *node);


/**
 * @return the nth child of this ASTNode or @c NULL if this node has no nth
 * child (n > ASTNode_getNumChildren() - 1).
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_getChild (const ASTNode_t *node, unsigned int n);

/**
 * @return the left child of this ASTNode.  This is equivalent to
 * ASTNode_getChild(node, 0);
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_getLeftChild (const ASTNode_t *node);

/**
 * @return the right child of this ASTNode or @c NULL if this node has no
 * right child.  If ASTNode_getNumChildren(node) > 1, then this is
 * equivalent to:
 *
 *   ASTNode_getChild(node, ASTNode_getNumChildren(node) - 1);
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_getRightChild (const ASTNode_t *node);

/**
 * @return the number of children of this ASTNode or 0 is this node has no
 * children.
 */
LIBSBML_EXTERN
unsigned int
ASTNode_getNumChildren (const ASTNode_t *node);


/**
 * Performs a depth-first search (DFS) of the tree rooted at node and
 * returns the List of nodes where predicate(node) returns @c true.
 *
 * The typedef for ASTNodePredicate is:
 *
 *   int (*ASTNodePredicate) (const ASTNode_t *node);
 *
 * where a return value of non-zero represents @c true and zero represents
 * @c false.
 *
 * The List returned is owned by the caller and should be freed with
 * List_free().  The ASTNodes in the list, however, are not owned by the
 * caller (as they still belong to the tree itself) and therefore should
 * not be freed.  That is, do not call List_freeItems().
 */
LIBSBML_EXTERN
List_t *
ASTNode_getListOfNodes (const ASTNode_t *node, ASTNodePredicate predicate);

/**
 * This method is identical in functionality to ASTNode_getListOfNodes(),
 * except the List is passed-in by the caller.
 */
LIBSBML_EXTERN
void
ASTNode_fillListOfNodes ( const ASTNode_t  *node,
                          ASTNodePredicate predicate,
                          List_t           *lst );


/**
 * @return the value of this ASTNode as a single character.  This function
 * should be called only when ASTNode_getType() is one of @c AST_PLUS,
 * @c AST_MINUS, @c AST_TIMES, @c AST_DIVIDE or @c AST_POWER.
 */
LIBSBML_EXTERN
char
ASTNode_getCharacter (const ASTNode_t *node);

/**
 * @return the value of this ASTNode as a (long) integer.  This function
 * should be called only when <code>ASTNode_getType() == AST_INTEGER</code>.
 */
LIBSBML_EXTERN
long
ASTNode_getInteger (const ASTNode_t *node);

/**
 * @return the value of this ASTNode as a string.  This function may be
 * called on nodes that are not operators (<code>ASTNode_isOperator(node)
 * == 0</code>) or numbers (<code>ASTNode_isNumber(node) == 0</code>).
 */
LIBSBML_EXTERN
const char *
ASTNode_getName (const ASTNode_t *node);

/**
 * @return the value of the numerator of this ASTNode.  This function
 * should be called only when ASTNode_getType() == AST_RATIONAL.
 */
LIBSBML_EXTERN
long
ASTNode_getNumerator (const ASTNode_t *node);

/**
 * @return the value of the denominator of this ASTNode.  This function
 * should be called only when <code>ASTNode_getType() ==
 * AST_RATIONAL</code>.
 */
LIBSBML_EXTERN
long
ASTNode_getDenominator (const ASTNode_t *node);

/**
 * @return the value of this ASTNode as a real (double).  This function
 * should be called only when ASTNode_isReal(node) != 0.
 *
 * This function performs the necessary arithmetic if the node type is @c
 * AST_REAL_E (<em>mantissa * 10<sup>exponent</sup></em>) or @c
 * AST_RATIONAL (<em>numerator / denominator</em>).
 */
LIBSBML_EXTERN
double
ASTNode_getReal (const ASTNode_t *node);

/**
 * @return the value of the mantissa of this ASTNode.  This function should
 * be called only when ASTNode_getType() is @c AST_REAL_E or @c AST_REAL.
 * If @c AST_REAL, this method is identical to ASTNode_getReal().
 */
LIBSBML_EXTERN
double
ASTNode_getMantissa (const ASTNode_t *node);

/**
 * @return the value of the exponent of this ASTNode.  This function should
 * be called only when ASTNode_getType() is @c AST_REAL_E or @c AST_REAL.
 */
LIBSBML_EXTERN
long
ASTNode_getExponent (const ASTNode_t *node);

/**
 * @return the precedence of this ASTNode (as defined in the SBML L1
 * specification).
 */
LIBSBML_EXTERN
int
ASTNode_getPrecedence (const ASTNode_t *node);

/**
 * @return the type of this ASTNode.
 */
LIBSBML_EXTERN
ASTNodeType_t
ASTNode_getType (const ASTNode_t *node);



LIBSBML_EXTERN
const char *
ASTNode_getId(const ASTNode_t * node);

LIBSBML_EXTERN
const char *
ASTNode_getClass(const ASTNode_t * node);

LIBSBML_EXTERN
const char *
ASTNode_getStyle(const ASTNode_t * node);


LIBSBML_EXTERN
const char *
ASTNode_getUnits(const ASTNode_t * node);


LIBSBML_EXTERN
int
ASTNode_isAvogadro (const ASTNode_t * node);


/**
 * @return true (non-zero) if this ASTNode is a boolean (a logical
 * operator, a relational operator, or the constants true or false), false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isBoolean (const ASTNode_t * node);

/**
 * @return true (non-zero) if this ASTNode returns a boolean, false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_returnsBoolean (const ASTNode_t *node);


/**
 * @return true (non-zero) if this ASTNode returns a boolean, false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_returnsBooleanForModel (const ASTNode_t *node, const Model_t* model);


/**
 * @return true (non-zero) if this ASTNode is a MathML constant (true,
 * false, pi, exponentiale), false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isConstant (const ASTNode_t * node);

/**
 * @return true (non-zero) if this ASTNode is a function in SBML L1, L2
 * (MathML) (everything from @c abs() to @c tanh()) or user-defined, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isFunction (const ASTNode_t * node);

/**
 * @return true if this ASTNode is the special IEEE 754 value infinity,
 * false otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isInfinity (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is of type @c AST_INTEGER, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isInteger (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is of type @c AST_LAMBDA, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isLambda (const ASTNode_t *node);

/**
 * @return true (non-zero) if the given ASTNode represents a log10()
 * function, false (0) otherwise.
 *
 * More precisley, the node type is @c AST_FUNCTION_LOG with two children
 * the first of which is an @c AST_INTEGER equal to 10.
 *
 * @if clike @see SBML_parseL3Formula()@endif@~
 * @if csharp @see SBML_parseL3Formula()@endif@~
 * @if python @see libsbml.parseL3Formula()@endif@~
 * @if java @see <code><a href="libsbml.html#parseL3Formula(String formula)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
 */
LIBSBML_EXTERN
int
ASTNode_isLog10 (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is a MathML logical operator
 * (and, or, not, xor), false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isLogical (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is a user-defined variable name
 * in SBML L1, L2 (MathML) or the special symbols delay or time, false (0)
 * otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isName (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is the special IEEE 754 value
 * not a number, false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isNaN (const ASTNode_t *node);

/**
 * @return true if this ASTNode is the special IEEE 754 value negative
 * infinity, false otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isNegInfinity (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is a number, false (0)
 * otherwise.
 *
 * This is functionally equivalent to:
 * @code
 *   ASTNode_isInteger(node) || ASTNode_isReal(node).
 * @endcode
 */
LIBSBML_EXTERN
int
ASTNode_isNumber (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is an operator, false (0)
 * otherwise.  Operators are: +, -, *, / and \^ (power).
 */
LIBSBML_EXTERN
int
ASTNode_isOperator (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is a piecewise function, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isPiecewise (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is of type @c AST_RATIONAL,
 * false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isRational (const ASTNode_t *node);

/**
 * @return true (non-zero) if the value of this ASTNode can represented as
 * a real number, false (0) otherwise.
 *
 * To be a represented as a real number, this node must be of one of the
 * following types: @c AST_REAL, @c AST_REAL_E or @c AST_RATIONAL.
 */
LIBSBML_EXTERN
int
ASTNode_isReal (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is a MathML relational operator
 * (==, >=, >, <=, < !=), false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isRelational (const ASTNode_t *node);

/**
 * @return true (non-zero) if the given ASTNode represents a sqrt()
 * function, false (0) otherwise.
 *
 * More precisley, the node type is @c AST_FUNCTION_ROOT with two children
 * the first of which is an @c AST_INTEGER equal to 2.
 */
LIBSBML_EXTERN
int
ASTNode_isSqrt (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is a unary minus, false (0)
 * otherwise.
 *
 * For numbers, unary minus nodes can be "collapsed" by negating the
 * number.  In fact, @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~
 * does this during its parse, and 
 * @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
 * has a configuration option that allows this behavior to be turned
 * on or off.  However, unary minus nodes for symbols (@c AST_NAMES) 
 * cannot be "collapsed", so this predicate function is necessary.
 *
 * A node is defined as a unary minus node if it is of type @c AST_MINUS
 * and has exactly one child.
 *
 * @if clike @see SBML_parseL3Formula()@endif@~
 * @if csharp @see SBML_parseL3Formula()@endif@~
 * @if python @see libsbml.parseL3Formula()@endif@~
 * @if java @see <code><a href="libsbml.html#parseL3Formula(String formula)">libsbml.parseL3Formula(String formula)</a></code>@endif@~
 */
LIBSBML_EXTERN
int
ASTNode_isUMinus (const ASTNode_t *node);

/**
 * @return true (non-zero) if this ASTNode is a unary plus, false (0)
 * otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isUPlus (const ASTNode_t *node);


/**
 * @return true (non-zero) if this ASTNode is of type @c AST_UNKNOWN, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isUnknown (const ASTNode_t *node);

LIBSBML_EXTERN
int
ASTNode_isSetId (const ASTNode_t *node);

LIBSBML_EXTERN
int
ASTNode_isSetClass (const ASTNode_t *node);

LIBSBML_EXTERN
int
ASTNode_isSetStyle (const ASTNode_t *node);


LIBSBML_EXTERN
int
ASTNode_isSetUnits (const ASTNode_t *node);


LIBSBML_EXTERN
int
ASTNode_hasUnits (const ASTNode_t *node);


/**
 * Sets the value of this ASTNode to the given character.  If character is
 * one of '+', '-', '*', '/' or '\^', the node type will be set accordingly.
 * For all other characters, the node type will be set to @c AST_UNKNOWN.
 */
LIBSBML_EXTERN
int
ASTNode_setCharacter (ASTNode_t *node, char value);

/**
 * Sets the value of this ASTNode to the given name.
 *
 * The node type will be set (to @c AST_NAME) ONLY IF the ASTNode was
 * previously an operator (ASTNode_isOperator(node) != 0) or number
 * (ASTNode_isNumber(node) != 0).  This allows names to be set for
 * @c AST_FUNCTIONs and the like.
 */
LIBSBML_EXTERN
int
ASTNode_setName (ASTNode_t *node, const char *name);

/**
 * Sets the value of this ASTNode to the given (long) integer and sets the
 * node type to @c AST_INTEGER.
 */
LIBSBML_EXTERN
int
ASTNode_setInteger (ASTNode_t *node, long value);

/**
 * Sets the value of this ASTNode to the given rational in two parts:
 * the numerator and denominator.  The node type is set to @c AST_RATIONAL.
 */
LIBSBML_EXTERN
int
ASTNode_setRational (ASTNode_t *node, long numerator, long denominator);

/**
 * Sets the value of this ASTNode to the given real (double) and sets the
 * node type to @c AST_REAL.
 *
 * This is functionally equivalent to:
 * @code
 *   ASTNode_setRealWithExponent(node, value, 0);
 * @endcode
 */
LIBSBML_EXTERN
int
ASTNode_setReal (ASTNode_t *node, double value);

/**
 * Sets the value of this ASTNode to the given real (double) in two parts:
 * the mantissa and the exponent.  The node type is set to @c AST_REAL_E.
 */
LIBSBML_EXTERN
int
ASTNode_setRealWithExponent (ASTNode_t *node, double mantissa, long exponent);

/**
 * Sets the type of this ASTNode to the given ASTNodeType_t value.
 */
LIBSBML_EXTERN
int
ASTNode_setType (ASTNode_t *node, ASTNodeType_t type);


LIBSBML_EXTERN
int
ASTNode_setId (ASTNode_t *node, const char *id);

LIBSBML_EXTERN
int
ASTNode_setClass (ASTNode_t *node, const char *className);

LIBSBML_EXTERN
int
ASTNode_setStyle (ASTNode_t *node, const char *style);


LIBSBML_EXTERN
int
ASTNode_setUnits (ASTNode_t *node, const char *units);

/**
 * Swap the children of this ASTNode with the children of that ASTNode.
 */
LIBSBML_EXTERN
int
ASTNode_swapChildren (ASTNode_t *node, ASTNode_t *that);


LIBSBML_EXTERN
int
ASTNode_unsetId (ASTNode_t *node);

LIBSBML_EXTERN
int
ASTNode_unsetClass (ASTNode_t *node);

LIBSBML_EXTERN
int
ASTNode_unsetStyle (ASTNode_t *node);


LIBSBML_EXTERN
int
ASTNode_unsetUnits (ASTNode_t *node);


LIBSBML_EXTERN
void
ASTNode_replaceArgument(ASTNode_t* node, const char * bvar, ASTNode_t* arg);

LIBSBML_EXTERN
void
ASTNode_reduceToBinary(ASTNode_t* node);


LIBSBML_EXTERN
SBase_t * 
ASTNode_getParentSBMLObject(ASTNode_t* node);

LIBSBML_EXTERN
int
ASTNode_removeChild(ASTNode_t* node, unsigned int n);

LIBSBML_EXTERN
int
ASTNode_replaceChild(ASTNode_t* node, unsigned int n, ASTNode_t * newChild);

LIBSBML_EXTERN
int
ASTNode_insertChild(ASTNode_t* node, unsigned int n, ASTNode_t * newChild);

LIBSBML_EXTERN
int
ASTNode_addSemanticsAnnotation(ASTNode_t* node, XMLNode_t * annotation);

LIBSBML_EXTERN
unsigned int
ASTNode_getNumSemanticsAnnotations(ASTNode_t* node);

LIBSBML_EXTERN
XMLNode_t *
ASTNode_getSemanticsAnnotation(ASTNode_t* node, unsigned int n);

LIBSBML_EXTERN
int 
ASTNode_setUserData(ASTNode_t* node, void *userData);

LIBSBML_EXTERN
void *
ASTNode_getUserData(ASTNode_t* node);

LIBSBML_EXTERN
int
ASTNode_hasCorrectNumberArguments(ASTNode_t* node);

LIBSBML_EXTERN
int
ASTNode_isWellFormedASTNode(ASTNode_t* node);

/** @cond doxygen-libsbml-internal */
LIBSBML_EXTERN
int
ASTNode_true(const ASTNode_t *node);
/** @endcond */


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* ASTNode_h */

