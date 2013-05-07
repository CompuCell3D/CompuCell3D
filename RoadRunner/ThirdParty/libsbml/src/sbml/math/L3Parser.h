/**
 * @file    L3Parser.h
 * @brief   Definition of the level 3 infix-to-mathml parser C functions.
 * @author  Lucian Smith
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

#ifndef L3Parser_h
#define L3Parser_h

#include <sbml/common/extern.h>
#include <sbml/math/ASTNode.h>

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


/**
 * Parses the given mathematical formula and returns a representation of it
 * as an Abstract Syntax Tree (AST).
 *
 * The text-string form of mathematical formulas read by this function
 * are expanded versions of the formats produced and read by @if clike SBML_formulaToString()@endif@if csharp SBML_formulaToString()@endif@if python libsbml.formulaToString()@endif@if java <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode)">libsbml.formulaToString(ASTNode tree)</a></code>@endif@~
 * and
 * @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~, 
 * respectively.  The latter two libSBML functions were originally
 * developed to support conversion between SBML Levels&nbsp;1 and&nbsp;2,
 * and were focused on the syntax of mathematical formulas used in SBML
 * Level&nbsp;1.  With time, and the use of MathML in SBML Levels&nbsp;2
 * and&nbsp;3, it became clear that supporting Level&nbsp;2 and&nbsp;3's
 * expanded mathematical syntax would be useful for software developers.
 * To maintain backwards compatibility, the original
 * @if clike SBML_formulaToString()@endif@if csharp SBML_formulaToString()@endif@if python libsbml.formulaToString()@endif@if java <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode)">libsbml.formulaToString(ASTNode tree)</a></code>@endif@~
 * and
 * @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~
 * have been left untouched, and instead, the new functionality is
 * provided in the form of
 * @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~.
 *
 * The following are the differences in the formula syntax supported by
 * this function, compared to what is supported by  @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~:
 *
 * @li Units may be asociated with bare numbers, using the following syntax:
 * <div style="margin: 10px auto 10px 25px; display: block">
 * <span class="code" style="background-color: #d0d0ee">number</span>
 * <span class="code" style="background-color: #edd">unit</span>
 * </div>
 * The <span class="code" style="background-color: #d0d0ee">number</span>
 * may be in any form (an integer, real, or rational
 * number), and the 
 * <span class="code" style="background-color: #edd">unit</span>
 * must conform to the syntax of an SBML identifier (technically, the
 * type defined as @c SId in the SBML specifications).  The whitespace between
 * <span class="code" style="background-color: #d0d0ee">number</span>
 * and <span class="code" style="background-color: #edd">unit</span>
 * is optional.
 * @li The Boolean function symbols @c &&, @c ||, @c !, and @c != may be used.
 * @li The @em modulo operation is allowed as the symbol @c @% and 
 * will produce a piecewise function in the MathML.
 * @li All inverse trigonometric functions may be defined in the infix
 * either using @c arc as a prefix or simply @c a; in other words, both
 * @c arccsc and @c acsc are interpreted as the operator @em arccosecant
 * defined in MathML.  (Many functions in the SBML Level&nbsp;1 infix-notation
 * parser implemented by @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~
 * are defined this way as well, but not all.)
 * @li The following expression is parsed as a rational number instead of
 * as a numerical division:
 * <pre style="display: block; margin-left: 25px">
 * (<span class="code" style="background-color: #d0d0ee">integer</span>/<span class="code" style="background-color: #d0d0ee">integer</span>)</pre>
 * No spaces are allowed in this construct; in other words,
 * &quot;<code>(3 / 4)</code>&quot; will be parsed into the MathML
 * <code>&lt;divide&gt;</code> construct rather than a rational number.  The 
 * general number syntax allows you to assign units to a rational number, e.g.,
 * &quot;<code>(3/4) ml</code>&quot;.  (If the string is a division, units
 * are not interpreted in this way.)
 * @li Various settings may be altered by using an L3ParserSettings object
 * in conjunction with the alternative function call
 * @if clike SBML_parseL3FormulaWithSettings()@endif@if csharp SBML_parseL3FormulaWithSettings()@endif@if python libsbml.parseL3FormulaWithSettings()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>@endif@~, including the following:
 * <ul>
 * <li> The function @c log with a single argument (&quot;<code>log(x)</code>&quot;) 
 * can be parsed as <code>log10(x)</code>, <code>ln(x)</code>, or treated
 * as an error, as desired.
 * <li> Unary minus signs can be collapsed or preserved; that is,
 * sequential pairs of unary minuses (e.g., &quot;<code>- -3</code>&quot;)
 * can be removed from the input entirely and single unary minuses can be
 * incorporated into the number node, or all minuses can be preserved in
 * the AST node structure.
 * <li> Parsing of units embedded in the input string can be turned on and
 * off.
 * <li> The string @c avogadro can be parsed as a MathML @em csymbol or
 * as an identifier.
 * <li> A Model object may optionally be provided to the parser using
 * the variant function call @if clike  SBML_parseL3FormulaWithModel()@endif@if csharp  SBML_parseL3FormulaWithModel()@endif@if python  libsbml.SBML_parseL3FormulaWithModel()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithModel(java.lang.String, org.sbml.libsbml.Model)">libsbml.parseL3FormulaWithModel(String formula, Model model)</a></code>@endif@~.
 * or stored in a L3ParserSettings object passed to the variant function
 * @if clike SBML_parseL3FormulaWithSettings()@endif@if csharp SBML_parseL3FormulaWithSettings()@endif@if python libsbml.parseL3FormulaWithSettings()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>@endif@~.
 * When a Model object is provided, identifiers (values of type @c SId)
 * from that model are used in preference to pre-defined MathML
 * definitions.  More precisely, the Model entities whose identifiers will
 * shadow identical symbols in the mathematical formula are: Species,
 * Compartment, Parameter, Reaction, and SpeciesReference.  For instance,
 * if the parser is given a Model containing a Species with the identifier
 * &quot;<code>pi</code>&quot;, and the formula to be parsed is
 * &quot;<code>3*pi</code>&quot;, the MathML produced will contain the
 * construct <code>&lt;ci&gt; pi &lt;/ci&gt;</code> instead of the
 * construct <code>&lt;pi/&gt;</code>.
 * <li> Similarly, when a Model object is provided, @c SId values of
 * user-defined functions present in the model will be used preferentially
 * over pre-defined MathML functions.  For example, if the passed-in Model
 * contains a FunctionDefinition with the identifier
 * &quot;<code>sin</code>&quot;, that function will be used instead of the
 * predefined MathML function <code>&lt;sin/&gt;</code>.
 * </ul>
 * These configuration settings cannot be changed using @em this function
 * (i.e., @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~), 
 * but they can be change on a per-call basis by using the alternative function 
 * @if clike SBML_parseL3FormulaWithSettings()@endif@if csharp SBML_parseL3FormulaWithSettings()@endif@if python libsbml.parseL3FormulaWithSettings()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>@endif@~
 *
 * This function returns the root node of the AST corresponding to the
 * formula given as the argument.  If the formula contains a syntax error,
 * this function will return @c NULL instead.  When @c NULL is returned, an
 * error is set; information about the error can be retrieved using
 * @if clike SBML_getLastParseL3Error()@endif@if csharp SBML_getLastParseL3Error()@endif@if python libsbml.getLastParseL3Error()@endif@if java <code><a href="libsbml.html#getLastParseL3Error()">libsbml.getLastParseL3Error()</a></code>@endif@~.
 *
 * Note that this facility and the SBML Level&nbsp;1-based @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~
 * are provided as a convenience by libSBML&mdash;the MathML standard does not
 * actually define a "string-form" equivalent to MathML expressions, so the
 * choice of formula syntax is arbitrary.  The approach taken by libSBML is
 * to start with the syntax defined by SBML Level&nbsp;1 (which in fact
 * used a text-string representation of formulas, and not MathML), and
 * expand it to include the above functionality.  This formula syntax is
 * based mostly on C programming syntax, and may contain operators,
 * function calls, symbols, and white space characters.  The following
 * table provides the precedence rules for the different entities that may
 * appear in formula strings.
 *
 * @htmlinclude math-precedence-table-l3.html
 * 
 * In the table above, @em operand implies the construct is an operand, @em
 * prefix implies the operation is applied to the following arguments, @em
 * unary implies there is one argument, and @em binary implies there are
 * two arguments.  The values in the <b>Precedence</b> column show how the
 * order of different types of operation are determined.  For example, the
 * expression <code>a + b * c</code> is evaluated as <code>a + (b * c)</code> 
 * because the @c * operator has higher precedence.  The
 * <b>Associates</b> column shows how the order of similar precedence
 * operations is determined; for example, <code>a && b || c</code> is
 * evaluated as <code>(a && b) || c</code> because the @c && and @c ||
 * operators are left-associative and have the same precedence.
 *
 * The function call syntax consists of a function name, followed by optional
 * white space, followed by an opening parenthesis token, followed by a
 * sequence of zero or more arguments separated by commas (with each comma
 * optionally preceded and/or followed by zero or more white space
 * characters), followed by a closing parenthesis token.  The function name
 * must be chosen from one of the pre-defined functions in SBML or a
 * user-defined function in the model.  The following table lists the names
 * of certain common mathematical functions; this table corresponds to
 * Table&nbsp;6 in the <a target="_blank" href="http://sbml.org/Documents/Specifications#SBML_Level_1_Version_2">SBML Level&nbsp;1 Version&nbsp;2 specification</a> with additions based on the 
 * functions added in SBML Level 2 and Level 3:
 *
 * @htmlinclude string-functions-table-l3.html
 *
 * Note that this function's interpretation of the string
 * &quot;<code>log</code>&quot; as a function with a single argument can be
 * changed; use the function @if clike SBML_parseL3FormulaWithSettings()@endif@if csharp SBML_parseL3FormulaWithSettings()@endif@if python libsbml.parseL3FormulaWithSettings()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>@endif@~
 * instead of this function and pass it an appropriate L3ParserSettings
 * object.  By default, unlike the SBML Level&nbsp;1 parser implemented by
 * @if clike SBML_parseFormula()@endif@if csharp SBML_parseFormula()@endif@if python libsbml.parseFormula()@endif@if java <code><a href="libsbml.html#parseFormula(java.lang.String)">libsbml.parseFormula(String formula)</a></code>@endif@~, 
 * the string &quot;<code>log</code>&quot; is interpreted as the base&nbsp;10
 * logarithm, and @em not as the natural logarithm.  However, you can change
 * the interpretation to be base-10 log, natural log, or as an error; since
 * the name "log" by itself is ambiguous, you require that the parser uses
 * @c log10 or @c ln instead, which are more clear.  Please refer to
 * @if clike SBML_parseL3FormulaWithSettings()@endif@if csharp SBML_parseL3FormulaWithSettings()@endif@if python libsbml.parseL3FormulaWithSettings()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>@endif@~.
 * 
 * In addition, the following symbols will be translated to their MathML
 * equivalents, if no symbol with the same @c SId identifier string exists
 * in the Model object provided:
 *
 * @htmlinclude string-values-table-l3.html
 * 
 * Note that whether the string &quot;<code>avogadro</code>&quot; is parsed
 * as an AST node of type @link ASTNodeType_t#AST_NAME_AVOGADRO
 * AST_NAME_AVOGADRO@endlink or @link ASTNodeType_t#AST_NAME
 * AST_NAME@endlink is configurable; use the alternate version of this
 * function, called
 * @if clike SBML_parseL3FormulaWithSettings()@endif@if csharp SBML_parseL3FormulaWithSettings()@endif@if python libsbml.parseL3FormulaWithSettings()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>@endif@~.
 * This functionality is provided because SBML Level&nbsp;2 models may not
 * use @link ASTNodeType_t#AST_NAME_AVOGADRO AST_NAME_AVOGADRO@endlink AST nodes.
 *
 * @param formula the text-string formula expression to be parsed
 *
 * @return the root node of an AST representing the mathematical formula,
 * or @c NULL if an error occurred while parsing the formula.  When @c NULL
 * is returned, an error is recorded internally; information about the
 * error can be retrieved using 
 * @if clike SBML_getLastParseL3Error()@endif@if csharp SBML_getLastParseL3Error()@endif@if python libsbml.getLastParseL3Error()@endif@if java <code><a href="libsbml.html#getLastParseL3Error()">libsbml.getLastParseL3Error()</a></code>@endif@~.
 *
 * @if clike @see SBML_formulaToString()
 * @see SBML_parseL3FormulaWithSettings()
 * @see SBML_parseL3Formula()
 * @see SBML_parseL3FormulaWithModel()
 * @see SBML_getLastParseL3Error()
 * @see SBML_getDefaultL3ParserSettings()
 * @endif@~
 * @if csharp @see SBML_formulaToString()
 * @see SBML_parseL3FormulaWithSettings()
 * @see SBML_parseL3Formula()
 * @see SBML_parseL3FormulaWithModel()
 * @see SBML_getLastParseL3Error()
 * @see SBML_getDefaultL3ParserSettings()
 * @endif@~
 * @if python @see libsbml.formulaToString()
 * @see libsbml.parseL3FormulaWithSettings()
 * @see libsbml.parseL3Formula()
 * @see libsbml.parseL3FormulaWithModel()
 * @see libsbml.getLastParseL3Error()
 * @see libsbml.getDefaultL3ParserSettings()
 * @endif@~
 * @if java @see <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode tree)">libsbml.formulaToString(ASTNode tree)</a></code>
 * @see <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>
 * @see <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>
 * @see <code><a href="libsbml.html#parseL3FormulaWithModel(java.lang.String, org.sbml.libsbml.Model)">parseL3FormulaWithModel(String formula, Model model)</a></code>
 * @see <code><a href="libsbml.html#getLastParseL3Error()">getLastParseL3Error()</a></code>
 * @see <code><a href="libsbml.html#getDefaultL3ParserSettings()">getDefaultL3ParserSettings()</a></code>
 * @endif@~
 */
LIBSBML_EXTERN
ASTNode_t *
SBML_parseL3Formula (const char *formula);


/**
 * Parses the given mathematical formula using specific a specific Model to
 * resolve symbols, and returns an Abstract Syntax Tree (AST)
 * representation of the result.
 *
 * This is identical to
 * @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~,
 * except that this function uses the given model in the argument @p model
 * to check against identifiers that appear in the @p formula.
 *
 * For more details about the parser, please see the definition of
 * the function @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~.
 *
 * @param formula the mathematical formula expression to be parsed
 *
 * @param model the Model object to use for checking identifiers
 *
 * @return the root node of an AST representing the mathematical formula,
 * or @c NULL if an error occurred while parsing the formula.  When @c NULL
 * is returned, an error is recorded internally; information about the
 * error can be retrieved using
 * @if clike SBML_getLastParseL3Error()@endif@if csharp SBML_getLastParseL3Error()@endif@if python libsbml.getLastParseL3Error()@endif@if java <code><a href="libsbml.html#getLastParseL3Error()">libsbml.getLastParseL3Error()</a></code>@endif@~.
 * 
 * @if clike @see SBML_formulaToString()
 * @see SBML_parseL3FormulaWithSettings()
 * @see SBML_parseL3Formula()
 * @see SBML_getLastParseL3Error()
 * @see SBML_getDefaultL3ParserSettings()
 * @endif@~
 * @if csharp @see SBML_formulaToString()
 * @see SBML_parseL3FormulaWithSettings()
 * @see SBML_parseL3Formula()
 * @see SBML_getLastParseL3Error()
 * @see SBML_getDefaultL3ParserSettings()
 * @endif@~
 * @if python @see libsbml.formulaToString()
 * @see libsbml.parseL3FormulaWithSettings()
 * @see libsbml.parseL3Formula()
 * @see libsbml.getLastParseL3Error()
 * @see libsbml.getDefaultL3ParserSettings()
 * @endif@~
 * @if java @see <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode tree)">libsbml.formulaToString(ASTNode tree)</a></code>
 * @see <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>
 * @see <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>
 * @see <code><a href="libsbml.html#getLastParseL3Error()">getLastParseL3Error()</a></code>
 * @see <code><a href="libsbml.html#getDefaultL3ParserSettings()">getDefaultL3ParserSettings()</a></code>
 * @endif@~
 */
LIBSBML_EXTERN
ASTNode_t *
SBML_parseL3FormulaWithModel (const char *formula, const Model_t * model);


/**
 * Parses the given mathematical formula using specific parser settings and
 * returns an Abstract Syntax Tree (AST) representation of the result.
 *
 * This is identical to
 @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~,
 * except that this function uses the parser settings given in the argument
 * @p settings.  The settings override the default parsing behavior.
 *
 * The parameter @p settings allows callers to change the following parsing
 * behaviors:
 *
 * @li Use a specific Model object against which identifiers to compare
 * identifiers.  This causes the parser to search the Model for identifiers
 * that the parser encounters in the formula.  If a given symbol in the
 * formula matches the identifier of a Species, Compartment, Parameter,
 * Reaction, SpeciesReference or FunctionDefinition in the Model, then the
 * symbol is assumed to refer to that model entity instead of any possible
 * mathematical terms with the same symbol.  For example, if the parser is
 * given a Model containing a Species with the identifier
 * &quot;<code>pi</code>&quot;, and the formula to be parsed is
 * &quot;<code>3*pi</code>&quot;, the MathML produced will contain the
 * construct <code>&lt;ci&gt; pi &lt;/ci&gt;</code> instead of the
 * construct <code>&lt;pi/&gt;</code>.
 * @li Whether to parse &quot;<code>log(x)</code>&quot; with a single
 * argument as the base 10
 * logarithm of x, the natural logarithm of x, or treat the case as an
 * error.
 * @li Whether to parse &quot;<code>number id</code>&quot; by interpreting
 * @c id as the identifier of a unit of measurement associated with the
 * number, or whether to treat the case as an error.
 * @li Whether to parse &quot;<code>avogadro</code>&quot; as an ASTNode of
 * type @link ASTNodeType_t#AST_NAME_AVOGADRO AST_NAME_AVOGADRO@endlink or
 * as type @link ASTNodeType_t#AST_NAME AST_NAME@endlink.
 * @li Whether to always create explicit ASTNodes of type @link
 * ASTNodeType_t#AST_MINUS AST_MINUS@endlink for all unary minuses, or
 * collapse and remove minuses where possible.
 *
 * For more details about the parser, please see the definition of
 * L3ParserSettings and
 * @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~.
 *
 * @param formula the mathematical formula expression to be parsed
 *
 * @param settings the settings to be used for this parser invocation
 *
 * @return the root node of an AST representing the mathematical formula,
 * or @c NULL if an error occurred while parsing the formula.  When @c NULL
 * is returned, an error is recorded internally; information about the
 * error can be retrieved using
 * @if clike SBML_getLastParseL3Error()@endif@if csharp SBML_getLastParseL3Error()@endif@if python libsbml.getLastParseL3Error()@endif@if java <code><a href="libsbml.html#getLastParseL3Error()">libsbml.getLastParseL3Error()</a></code>@endif@~.
 * 
 * @if clike @see SBML_formulaToString()
 * @see SBML_parseL3Formula()
 * @see SBML_parseL3FormulaWithModel()
 * @see SBML_getLastParseL3Error()
 * @see SBML_getDefaultL3ParserSettings()
 * @endif@~
 * @if csharp @see SBML_formulaToString()
 * @see SBML_parseL3Formula()
 * @see SBML_parseL3FormulaWithModel()
 * @see SBML_getLastParseL3Error()
 * @see SBML_getDefaultL3ParserSettings()
 * @endif@~
 * @if python @see libsbml.formulaToString()
 * @see libsbml.parseL3Formula()
 * @see libsbml.parseL3FormulaWithModel()
 * @see libsbml.getLastParseL3Error()
 * @see libsbml.getDefaultL3ParserSettings()
 * @endif@~
 * @if java @see <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode tree)">libsbml.formulaToString(ASTNode tree)</a></code>
 * @see <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>
 * @see <code><a href="libsbml.html#parseL3FormulaWithModel(java.lang.String, org.sbml.libsbml.Model)">parseL3FormulaWithModel(String formula, Model model)</a></code>
 * @see <code><a href="libsbml.html#getLastParseL3Error()">getLastParseL3Error()</a></code>
 * @see <code><a href="libsbml.html#getDefaultL3ParserSettings()">getDefaultL3ParserSettings()</a></code>
 * @endif@~
 */
LIBSBML_EXTERN
ASTNode_t *
SBML_parseL3FormulaWithSettings (const char *formula, const L3ParserSettings_t *settings);


/**
 * Returns a copy of the default parser settings used by @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~.
 * 
 * The settings structure allows callers to change the following parsing
 * behaviors:
 * 
 * @li Use a specific Model object against which identifiers to compare
 * identifiers.  This causes the parser to search the Model for identifiers
 * that the parser encounters in the formula.  If a given symbol in the
 * formula matches the identifier of a Species, Compartment, Parameter,
 * Reaction, SpeciesReference or FunctionDefinition in the Model, then the
 * symbol is assumed to refer to that model entity instead of any possible
 * mathematical terms with the same symbol.  For example, if the parser is
 * given a Model containing a Species with the identifier
 * &quot;<code>pi</code>&quot;, and the formula to be parsed is
 * &quot;<code>3*pi</code>&quot;, the MathML produced will contain the
 * construct <code>&lt;ci&gt; pi &lt;/ci&gt;</code> instead of the
 * construct <code>&lt;pi/&gt;</code>.
 * @li Whether to parse &quot;<code>log(x)</code>&quot; with a single
 * argument as the base 10
 * logarithm of x, the natural logarithm of x, or treat the case as an
 * error.
 * @li Whether to parse &quot;<code>number id</code>&quot; by interpreting
 * @c id as the identifier of a unit of measurement associated with the
 * number, or whether to treat the case as an error.
 * @li Whether to parse &quot;<code>avogadro</code>&quot; as an ASTNode of
 * type @link ASTNodeType_t#AST_NAME_AVOGADRO AST_NAME_AVOGADRO@endlink or
 * as type @link ASTNodeType_t#AST_NAME AST_NAME@endlink.
 * @li Whether to always create explicit ASTNodes of type @link
 * ASTNodeType_t#AST_MINUS AST_MINUS@endlink for all unary minuses, or
 * collapse and remove minuses where possible.
 *
 * For more details about the parser, please see the definition of
 * L3ParserSettings and
 * @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~.
 * 
 * @if clike @see SBML_formulaToString()
 * @see SBML_parseL3FormulaWithSettings()
 * @see SBML_parseL3Formula()
 * @see SBML_parseL3FormulaWithModel()
 * @see SBML_getLastParseL3Error()
 * @endif@~
 * @if csharp @see SBML_formulaToString()
 * @see SBML_parseL3FormulaWithSettings()
 * @see SBML_parseL3Formula()
 * @see SBML_parseL3FormulaWithModel()
 * @see SBML_getLastParseL3Error()
 * @endif@~
 * @if python @see libsbml.formulaToString()
 * @see libsbml.parseL3FormulaWithSettings()
 * @see libsbml.parseL3Formula()
 * @see libsbml.parseL3FormulaWithModel()
 * @see libsbml.getLastParseL3Error()
 * @endif@~
 * @if java @see <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode tree)">libsbml.formulaToString(ASTNode tree)</a></code>
 * @see <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>
 * @see <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>
 * @see <code><a href="libsbml.html#parseL3FormulaWithModel(java.lang.String, org.sbml.libsbml.Model)">parseL3FormulaWithModel(String formula, Model model)</a></code>
 * @see <code><a href="libsbml.html#getLastParseL3Error()">getLastParseL3Error()</a></code>
 * @endif@~
 */
LIBSBML_EXTERN
L3ParserSettings_t*
SBML_getDefaultL3ParserSettings ();


/**
 * Returns the last error reported by the parser.
 *
 * If @if clike SBML_parseL3Formula()@endif@if csharp SBML_parseL3Formula()@endif@if python libsbml.parseL3Formula()@endif@if java <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>@endif@~, 
 * @if clike SBML_parseL3FormulaWithSettings()@endif@if csharp SBML_parseL3FormulaWithSettings()@endif@if python libsbml.parseL3FormulaWithSettings()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>@endif@~, or
 * @if clike SBML_parseL3FormulaWithModel()@endif@if csharp SBML_parseL3FormulaWithModel()@endif@if python libsbml.parseL3FormulaWithModel()@endif@if java <code><a href="libsbml.html#parseL3FormulaWithModel(java.lang.String, org.sbml.libsbml.Model)">libsbml.parseL3FormulaWithModel(String formula, Model model)</a></code>@endif@~ return @c NULL, an error is set internally which is accessible
 * via this function. 
 *
 * @return a string describing the error that occurred.  This will contain
 * the string the parser was trying to parse, which character it had parsed
 * when it encountered the error, and a description of the error.
 *
 * @if clike @see SBML_formulaToString()
 * @see SBML_parseL3FormulaWithSettings()
 * @see SBML_parseL3Formula()
 * @see SBML_parseL3FormulaWithModel()
 * @see SBML_getDefaultL3ParserSettings()
 * @endif@~
 * @if csharp @see SBML_formulaToString()
 * @see SBML_parseL3FormulaWithSettings()
 * @see SBML_parseL3Formula()
 * @see SBML_parseL3FormulaWithModel()
 * @see SBML_getDefaultL3ParserSettings()
 * @endif@~
 * @if python @see libsbml.formulaToString()
 * @see libsbml.parseL3FormulaWithSettings()
 * @see libsbml.parseL3Formula()
 * @see libsbml.parseL3FormulaWithModel()
 * @see libsbml.getDefaultL3ParserSettings()
 * @endif@~
 * @if java @see <code><a href="libsbml.html#formulaToString(org.sbml.libsbml.ASTNode tree)">libsbml.formulaToString(ASTNode tree)</a></code>
 * @see <code><a href="libsbml.html#parseL3FormulaWithSettings(java.lang.String, org.sbml.libsbml.L3ParserSettings)">libsbml.parseL3FormulaWithSettings(String formula, L3ParserSettings settings)</a></code>
 * @see <code><a href="libsbml.html#parseL3Formula(java.lang.String)">libsbml.parseL3Formula(String formula)</a></code>
 * @see <code><a href="libsbml.html#parseL3FormulaWithModel(java.lang.String, org.sbml.libsbml.Model)">parseL3FormulaWithModel(String formula, Model model)</a></code>
 * @see <code><a href="libsbml.html#getDefaultL3ParserSettings()">getDefaultL3ParserSettings()</a></code>
 * @endif@~
 */
LIBSBML_EXTERN
char*
SBML_getLastParseL3Error();

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END
#endif /* L3Parser_h */
