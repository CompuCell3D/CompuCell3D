/**
 * @file    XMLNode.h
 * @brief   Class definition of XMLNode, a node in an XML document tree.
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
 * ---------------------------------------------------------------------- -->
 *
 * @class XMLNode
 * @brief Representation of a node in an XML document tree.
 * 
 * Beginning with version 3.0.0, libSBML implements an XML abstraction
 * layer.  This layer presents a uniform XML interface to calling programs
 * regardless of which underlying XML parser libSBML has actually been
 * configured to use.  The basic data object in the XML abstraction is a
 * @em node, represented by XMLNode.
 *
 * An XMLNode can contain any number of children.  Each child is another
 * XMLNode, thereby forming a tree.  The methods XMLNode::getNumChildren()
 * and XMLNode::getChild(@if java long n@endif) can be used to access the tree
 * structure starting from a given node.
 *
 * Each XMLNode is subclassed from XMLToken, and thus has the same methods
 * available as XMLToken.  These methods include XMLToken::getNamespaces(),
 * XMLToken::getPrefix(), XMLToken::getName(), XMLToken::getURI(), and
 * XMLToken::getAttributes().
 *
 * @section xmlnode-str2xmlnode Conversion between an XML string and an XMLNode
 *
 * LibSBML provides the following utility functions for converting an XML
 * string (e.g., <code>&lt;annotation&gt;...&lt;/annotation&gt;</code>)
 * to/from an XMLNode object.
 * <ul>
 * <li> XMLNode::toXMLString() returns a string representation of the XMLNode object. 
 *
 * <li> XMLNode::convertXMLNodeToString(@if java XMLNode node@endif)
 * (static function) returns a string representation 
 * of the given XMLNode object.
 *
 * <li> XMLNode::convertStringToXMLNode(@if java String xml@endif)
 * (static function) returns an XMLNode object converted 
 * from the given XML string.
 * </ul>
 *
 * The returned XMLNode object by XMLNode::convertStringToXMLNode(@if java String xml@endif)
 * is a dummy root (container) XMLNode if the given XML string has two or
 * more top-level elements (e.g.,
 * &quot;<code>&lt;p&gt;...&lt;/p&gt;&lt;p&gt;...&lt;/p&gt;</code>&quot;). In the
 * dummy root node, each top-level element in the given XML string is
 * contained as a child XMLNode. XMLToken::isEOF() can be used to identify
 * if the returned XMLNode object is a dummy node or not.  Here is an
 * example: @if clike
 * @verbatim
// Checks if the XMLNode object returned by XMLNode::convertStringToXMLNode() is a dummy root node:
                                                                                         
std::string str = "..."; 
XMLNode* xn = XMLNode::convertStringToXMLNode(str);                                      
if ( xn == NULL )
{                                                                                      
  // returned value is null (error)                                                    
  ...
}                                                                                      
else if ( xn->isEOF() )                                                                 
{                                                                                      
  // root node is a dummy node                                                         
  for ( int i = 0; i < xn->getNumChildren(); i++ )                                          
  {                                                                                    
    // access to each child node of the dummy node.                                    
    XMLNode& xnChild = xn->getChild(i);                                                  
    ...                                                                                
  }                                                                                    
}                                                                                      
else                                                                                   
{                                                                                      
  // root node is NOT a dummy node                                                     
  ...                                                                                  
}
@endverbatim
  *  @endif@if java
@verbatim
// Checks if the returned XMLNode object is a dummy root node:

String str = "...";
XMLNode xn = XMLNode.convertStringToXMLNode(str);
if ( xn == null )
{
  // returned value is null (error)
  ...
}
else if ( xn.isEOF() )
{
  // root node is a dummy node
  for ( int i = 0; i < xn.getNumChildren(); i++ )
  {
    // access to each child node of the dummy node.
    XMLNode xnChild = xn.getChild(i);
    ...
  }
}
else
{
  // root node is NOT a dummy node
  ...
}
@endverbatim
 * @endif@if python
@verbatim
xn = XMLNode.convertStringToXMLNode("<p></p>")
if xn == None:
  # Do something to handle exceptional situation.

elif xn.isEOF():
  # Node is a dummy node.

else:
  # None is not a dummy node.
@endverbatim
 * @endif@~
 */

#ifndef XMLNode_h
#define XMLNode_h

#include <sbml/xml/XMLExtern.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/common/sbmlfwd.h>


#ifdef __cplusplus

#include <vector>
#include <cstdlib>

LIBSBML_CPP_NAMESPACE_BEGIN

/** @cond doxygen-libsbml-internal */
class XMLInputStream;
class XMLOutputStream;
/** @endcond */


class LIBLAX_EXTERN XMLNode : public XMLToken
{
public:

  /**
   * Creates a new empty XMLNode with no children.
   */
  XMLNode ();


  /**
   * Creates a new XMLNode by copying token.
   *
   * @param token XMLToken to be copied to XMLNode
   */
  XMLNode (const XMLToken& token);

  /**
   * Creates a new start element XMLNode with the given set of attributes and
   * namespace declarations.
   *
   * @param triple XMLTriple.
   * @param attributes XMLAttributes, the attributes to set.
   * @param namespaces XMLNamespaces, the namespaces to set.
   * @param line an unsigned int, the line number (default = 0).
   * @param column an unsigned int, the column number (default = 0).
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  XMLNode (  const XMLTriple&     triple
           , const XMLAttributes& attributes
           , const XMLNamespaces& namespaces
           , const unsigned int   line   = 0
           , const unsigned int   column = 0 );


  /**
   * Creates a start element XMLNode with the given set of attributes.
   *
   * @param triple XMLTriple.
   * @param attributes XMLAttributes, the attributes to set.
   * @param line an unsigned int, the line number (default = 0).
   * @param column an unsigned int, the column number (default = 0).
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
  */
  XMLNode (  const XMLTriple&      triple
           , const XMLAttributes&  attributes
           , const unsigned int    line   = 0
           , const unsigned int    column = 0 );


  /**
   * Creates an end element XMLNode.
   *
   * @param triple XMLTriple.
   * @param line an unsigned int, the line number (default = 0).
   * @param column an unsigned int, the column number (default = 0).
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  XMLNode (  const XMLTriple&    triple
           , const unsigned int  line   = 0
           , const unsigned int  column = 0 );


  /**
   * Creates a text XMLNode.
   *
   * @param chars a string, the text to be added to the XMLToken
   * @param line an unsigned int, the line number (default = 0).
   * @param column an unsigned int, the column number (default = 0).
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  XMLNode (  const std::string&  chars
           , const unsigned int  line   = 0
           , const unsigned int  column = 0 );


  /** @cond doxygen-libsbml-internal */
  /**
   * Creates a new XMLNode by reading XMLTokens from stream.  
   *
   * The stream must be positioned on a start element
   * (<code>stream.peek().isStart() == true</code>) and will be read until
   * the matching end element is found.
   *
   * @param stream XMLInputStream from which XMLNode is to be created.
   */
  XMLNode (XMLInputStream& stream);
  /** @endcond */


  /**
   * Destroys this XMLNode.
   */
  virtual ~XMLNode ();

  
  /**
   * Copy constructor; creates a copy of this XMLNode.
   * 
   * @param orig the XMLNode instance to copy.
   *
   * @throws @if python ValueError @else XMLConstructorException @endif@~
   * Thrown if the argument @p orig is @c NULL.
   */
  XMLNode(const XMLNode& orig);


  /**
   * Assignment operator for XMLNode.
   *
   * @param rhs The XMLNode object whose values are used as the basis
   * of the assignment.
   *
   * @throws @if python ValueError @else XMLConstructorException @endif@~
   * Thrown if the argument @p rhs is @c NULL.
   */
  XMLNode& operator=(const XMLNode& rhs);


  /**
   * Creates and returns a deep copy of this XMLNode.
   * 
   * @return a (deep) copy of this XMLNode.
   */
  XMLNode* clone () const;


  /**
   * Adds a copy of @p node as a child of this XMLNode.
   *
   * The given @p node is added at the end of the list of children.
   *
   * @param node the XMLNode to be added as child.
   *
   * @return integer value indicating success/failure of the
   * function.   The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_INVALID_XML_OPERATION LIBSBML_INVALID_XML_OPERATION @endlink
   *
   * @note The given node is added at the end of the children list.
   */
  int addChild (const XMLNode& node);


  /**
   * Inserts a copy of the given node as the <code>n</code>th child of this
   * XMLNode.
   *
   * If the given index @p n is out of range for this XMLNode instance,
   * the @p node is added at the end of the list of children.  Even in
   * that situation, this method does not throw an error.
   *
   * @param n an integer, the index at which the given node is inserted
   * @param node an XMLNode to be inserted as <code>n</code>th child.
   *
   * @return a reference to the newly-inserted child @p node
   */
  XMLNode& insertChild (unsigned int n, const XMLNode& node);


  /**
   * Removes the <code>n</code>th child of this XMLNode and returns the
   * removed node.
   *
   * It is important to keep in mind that a given XMLNode may have more
   * than one child.  Calling this method erases all existing references to
   * child nodes @em after the given position @p n.  If the index @p n is
   * greater than the number of child nodes in this XMLNode, this method
   * takes no action (and returns @c NULL).
   *
   * @param n an integer, the index of the node to be removed
   *
   * @return the removed child, or @c NULL if @p n is greater than the number
   * of children in this node
   *
   * @note The caller owns the returned node and is responsible for deleting it.
   */
  XMLNode* removeChild(unsigned int n);


  /**
   * Removes all children from this node.
   * @return integer value indicating success/failure of the
   * function.   The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   */
  int removeChildren();


  /**
   * Returns the <code>n</code>th child of this XMLNode.
   *
   * If the index @p n is greater than the number of child nodes, or it is
   * 0 or less, this method returns an empty node.
   *
   * @param n the index of the node to return
   * 
   * @return the  <code>n</code>th child of this XMLNode.
   */
  XMLNode& getChild (unsigned int n);


  /**
   * Returns the  <code>n</code>th child of this XMLNode.
   *
   * If the index @p n is greater than the number of child nodes, or it is
   * 0 or less, this method returns an empty node.
   *
   * @param n an integer, the index of the node to return
   * 
   * @return the  <code>n</code>th child of this XMLNode.
   */
  const XMLNode& getChild (unsigned int n) const;

	/**
	 * Returns the first child of this XMLNode with the corresponding name.
	 *
	 * If no child with corrsponding name can be found, 
	 * this method returns an empty node.
	 *
	 * @param name the name of the node to return
	 * 
	 * @return the first child of this XMLNode with given name.
	 */
	XMLNode& getChild (const std::string&  name);	

	/**
	 * Returns the first child of this XMLNode with the corresponding name.
	 *
	 * If no child with corrsponding name can be found, 
	 * this method returns an empty node.
	 *
	 * @param name the name of the node to return
	 * 
	 * @return the first child of this XMLNode with given name.
	 */
	const XMLNode& getChild (const std::string&  name) const;
	
	/**
	 * Return the index of the first child of this XMLNode with the given name.
	 *
	 *
	 * @param name a string, the name of the child for which the 
	 * index is required.
	 *
	 * @return the index of the first child of this XMLNode with the given name, or -1 if not present.
	 */
	int getIndex (const std::string& name) const;

	/**
	 * Return a boolean indicating whether this XMLNode has a child with the given name.
	 *
	 *
	 * @param name a string, the name of the child to be checked.
	 *
	 * @return boolean indicating whether this XMLNode has a child with the given name.
	 */
	bool hasChild (const std::string& name) const;
	
	/**
	 * Compare this XMLNode against another XMLNode returning true if both nodes
	 * represent the same XML tree, or false otherwise.
	 *
	 *
	 * @param other another XMLNode to compare against
	 *
	 * @return boolean indicating whether this XMLNode represents the same XML tree as another.
	 */
	bool equals(const XMLNode& other) const;
	

  /**
   * Returns the number of children for this XMLNode.
   *
   * @return the number of children for this XMLNode.
   */
  unsigned int getNumChildren () const;


  /** @cond doxygen-libsbml-internal */
  /**
   * Writes this XMLNode and its children to stream.
   *
   * @param stream XMLOutputStream, stream to which this XMLNode
   * is to be written.
   */
  void write (XMLOutputStream& stream) const;
  /** @endcond */


  /**
   * Returns a string representation of this XMLNode. 
   *
   * @return a string derived from this XMLNode.
   */
  std::string toXMLString() const;


  /**
   * Returns a string representation of a given XMLNode. 
   *
   * @param node the XMLNode to be represented as a string
   *
   * @return a string-form representation of @p node
   */
  static std::string convertXMLNodeToString(const XMLNode* node);


  /**
   * Returns an XMLNode which is derived from a string containing XML
   * content.
   *
   * The XML namespace must be defined using argument @p xmlns if the
   * corresponding XML namespace attribute is not part of the string of the
   * first argument.
   *
   * @param xmlstr string to be converted to a XML node.
   * @param xmlns XMLNamespaces the namespaces to set (default value is @c NULL).
   *
   * @note The caller owns the returned XMLNode and is reponsible for
   * deleting it.  The returned XMLNode object is a dummy root (container)
   * XMLNode if the top-level element in the given XML string is NOT
   * <code>&lt;html&gt;</code>, <code>&lt;body&gt;</code>,
   * <code>&lt;annotation&gt;</code>, or <code>&lt;notes&gt;</code>.  In
   * the dummy root node, each top-level element in the given XML string is
   * contained as a child XMLNode. XMLToken::isEOF() can be used to
   * identify if the returned XMLNode object is a dummy node.
   *
   * @return a XMLNode which is converted from string @p xmlstr.  If the
   * conversion failed, this method returns @c NULL.
   *
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  static XMLNode* convertStringToXMLNode(const std::string& xmlstr,
                                         const XMLNamespaces* xmlns = NULL);


#ifndef SWIG

  /** @cond doxygen-libsbml-internal */
  /**
   * Inserts this XMLNode and its children into stream.
   *
   * @param stream XMLOutputStream, stream to which the XMLNode
   * is to be written.
   * @param node XMLNode, node to be written to stream.
   *
   * @return the stream with the node inserted.
   */
  LIBLAX_EXTERN
  friend
  XMLOutputStream& operator<< (XMLOutputStream& stream, const XMLNode& node);
  /** @endcond */

#endif  /* !SWIG */


protected:
  /** @cond doxygen-libsbml-internal */

  std::vector<XMLNode> mChildren;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS

/* ----------------------------------------------------------------------------
 * See the .cpp file for the documentation of the following functions.
 * --------------------------------------------------------------------------*/


LIBLAX_EXTERN
XMLNode_t *
XMLNode_create (void);


LIBLAX_EXTERN
XMLNode_t *
XMLNode_createFromToken (const XMLToken_t *token);


LIBLAX_EXTERN
XMLNode_t *
XMLNode_createStartElement  (const XMLTriple_t *triple,
			     const XMLAttributes_t *attr);


LIBLAX_EXTERN
XMLNode_t *
XMLNode_createStartElementNS (const XMLTriple_t *triple,
			      const XMLAttributes_t *attr,
			      const XMLNamespaces_t *ns);


LIBLAX_EXTERN
XMLNode_t *
XMLNode_createEndElement (const XMLTriple_t *triple);


LIBLAX_EXTERN
XMLNode_t *
XMLNode_createTextNode (const char *text);


LIBLAX_EXTERN
void
XMLNode_free (XMLNode_t *node);


LIBLAX_EXTERN
XMLNode_t *
XMLNode_clone (const XMLNode_t* c);


LIBLAX_EXTERN
int
XMLNode_addChild (XMLNode_t *node, const XMLNode_t *child);


LIBLAX_EXTERN
XMLNode_t*
XMLNode_insertChild (XMLNode_t *node, unsigned int n, const XMLNode_t *child);


LIBLAX_EXTERN
XMLNode_t* 
XMLNode_removeChild(XMLNode_t *node, unsigned int n);


LIBLAX_EXTERN
int
XMLNode_removeChildren (XMLNode_t *node);


LIBLAX_EXTERN
const char *
XMLNode_getCharacters (const XMLNode_t *node);


LIBLAX_EXTERN
const char *
XMLNode_getName (const XMLNode_t *node);


LIBLAX_EXTERN
int 
XMLNode_setTriple(XMLNode_t *node, const XMLTriple_t *triple);


LIBLAX_EXTERN
const char *
XMLNode_getPrefix (const XMLNode_t *node);


LIBLAX_EXTERN
const char *
XMLNode_getURI (const XMLNode_t *node);



LIBLAX_EXTERN
const XMLAttributes_t *
XMLNode_getAttributes (const XMLNode_t *node);


LIBLAX_EXTERN
int 
XMLNode_setAttributes (XMLNode_t *node, const XMLAttributes_t* attributes);


LIBLAX_EXTERN
int 
XMLNode_addAttr ( XMLNode_t *node,  const char* name, const char* value );
		   

LIBLAX_EXTERN
int 
XMLNode_addAttrWithNS ( XMLNode_t *node,  const char* name
	                , const char* value
    	                , const char* namespaceURI
	                , const char* prefix      );


LIBLAX_EXTERN
int 
XMLNode_addAttrWithTriple (XMLNode_t *node, const XMLTriple_t *triple, const char* value);


LIBLAX_EXTERN
int 
XMLNode_removeAttr (XMLNode_t *node, int n);


LIBLAX_EXTERN
int 
XMLNode_removeAttrByName (XMLNode_t *node, const char* name);


LIBLAX_EXTERN
int 
XMLNode_removeAttrByNS (XMLNode_t *node, const char* name, const char* uri);


LIBLAX_EXTERN
int 
XMLNode_removeAttrByTriple (XMLNode_t *node, const XMLTriple_t *triple);


LIBLAX_EXTERN
int 
XMLNode_clearAttributes(XMLNode_t *node);


LIBLAX_EXTERN
int 
XMLNode_getAttrIndex (const XMLNode_t *node, const char* name, const char* uri);


LIBLAX_EXTERN
int 
XMLNode_getAttrIndexByTriple (const XMLNode_t *node, const XMLTriple_t *triple);


LIBLAX_EXTERN
int 
XMLNode_getAttributesLength (const XMLNode_t *node);


LIBLAX_EXTERN
char* 
XMLNode_getAttrName (const XMLNode_t *node, int index);


LIBLAX_EXTERN
char* 
XMLNode_getAttrPrefix (const XMLNode_t *node, int index);


LIBLAX_EXTERN
char* 
XMLNode_getAttrPrefixedName (const XMLNode_t *node, int index);


LIBLAX_EXTERN
char* 
XMLNode_getAttrURI (const XMLNode_t *node, int index);


LIBLAX_EXTERN
char* 
XMLNode_getAttrValue (const XMLNode_t *node, int index);


LIBLAX_EXTERN
char* 
XMLNode_getAttrValueByName (const XMLNode_t *node, const char* name);


LIBLAX_EXTERN
char* 
XMLNode_getAttrValueByNS (const XMLNode_t *node, const char* name, const char* uri);


LIBLAX_EXTERN
char* 
XMLNode_getAttrValueByTriple (const XMLNode_t *node, const XMLTriple_t *triple);


LIBLAX_EXTERN
int
XMLNode_hasAttr (const XMLNode_t *node, int index);


LIBLAX_EXTERN
int
XMLNode_hasAttrWithName (const XMLNode_t *node, const char* name);

LIBLAX_EXTERN
int
XMLNode_hasAttrWithNS (const XMLNode_t *node, const char* name, const char* uri);


LIBLAX_EXTERN
int
XMLNode_hasAttrWithTriple (const XMLNode_t *node, const XMLTriple_t *triple);


LIBLAX_EXTERN
int
XMLNode_isAttributesEmpty (const XMLNode_t *node);



LIBLAX_EXTERN
const XMLNamespaces_t *
XMLNode_getNamespaces (const XMLNode_t *node);


LIBLAX_EXTERN
int 
XMLNode_setNamespaces(XMLNode_t *node, const XMLNamespaces_t* namespaces);


LIBLAX_EXTERN
int 
XMLNode_addNamespace (XMLNode_t *node, const char* uri, const char* prefix);


LIBLAX_EXTERN
int 
XMLNode_removeNamespace (XMLNode_t *node, int index);


LIBLAX_EXTERN
int 
XMLNode_removeNamespaceByPrefix (XMLNode_t *node, const char* prefix);


LIBLAX_EXTERN
int 
XMLNode_clearNamespaces (XMLNode_t *node);


LIBLAX_EXTERN
int 
XMLNode_getNamespaceIndex (const XMLNode_t *node, const char* uri);


LIBLAX_EXTERN
int 
XMLNode_getNamespaceIndexByPrefix (const XMLNode_t *node, const char* prefix);


LIBLAX_EXTERN
int 
XMLNode_getNamespacesLength (const XMLNode_t *node);


LIBLAX_EXTERN
char* 
XMLNode_getNamespacePrefix (const XMLNode_t *node, int index);


LIBLAX_EXTERN
char* 
XMLNode_getNamespacePrefixByURI (const XMLNode_t *node, const char* uri);


LIBLAX_EXTERN
char* 
XMLNode_getNamespaceURI (const XMLNode_t *node, int index);


LIBLAX_EXTERN
char* 
XMLNode_getNamespaceURIByPrefix (const XMLNode_t *node, const char* prefix);


LIBLAX_EXTERN
int
XMLNode_isNamespacesEmpty (const XMLNode_t *node);


LIBLAX_EXTERN
int
XMLNode_hasNamespaceURI(const XMLNode_t *node, const char* uri);


LIBLAX_EXTERN
int
XMLNode_hasNamespacePrefix(const XMLNode_t *node, const char* prefix);


LIBLAX_EXTERN
int
XMLNode_hasNamespaceNS(const XMLNode_t *node, const char* uri, const char* prefix);
                        


LIBLAX_EXTERN
int
XMLNode_isElement (const XMLNode_t *node);


LIBLAX_EXTERN
int
XMLNode_isEnd (const XMLNode_t *node); 


LIBLAX_EXTERN
int
XMLNode_isEndFor (const XMLNode_t *node, const XMLNode_t *element);


LIBLAX_EXTERN
int
XMLNode_isEOF (const XMLNode_t *node);


LIBLAX_EXTERN
int
XMLNode_isStart (const XMLNode_t *node);


LIBLAX_EXTERN
int
XMLNode_isText (const XMLNode_t *node);


LIBLAX_EXTERN
int
XMLNode_setEnd (XMLNode_t *node);


LIBLAX_EXTERN
int
XMLNode_setEOF (XMLNode_t *node);


LIBLAX_EXTERN
int
XMLNode_unsetEnd (XMLNode_t *node);


LIBLAX_EXTERN
const XMLNode_t *
XMLNode_getChild (const XMLNode_t *node, const int n);


LIBLAX_EXTERN
XMLNode_t *
XMLNode_getChildNC (XMLNode_t *node, const unsigned int n);

LIBLAX_EXTERN
XMLNode_t *
XMLNode_getChildForNameNC (XMLNode_t *node, const char*  name);	

LIBLAX_EXTERN
const XMLNode_t *
XMLNode_getChildForName (const XMLNode_t *node, const char*  name);

LIBLAX_EXTERN
int 
XMLNode_getIndex (const XMLNode_t *node, const char*  name);

LIBLAX_EXTERN
int 
XMLNode_hasChild (const XMLNode_t *node, const char*  name);

LIBLAX_EXTERN
int 
XMLNode_equals(const XMLNode_t *node, const XMLNode_t* other);

LIBLAX_EXTERN
unsigned int
XMLNode_getNumChildren (const XMLNode_t *node);


LIBLAX_EXTERN
char *
XMLNode_toXMLString(const XMLNode_t *node);


LIBLAX_EXTERN
const char *
XMLNode_convertXMLNodeToString(const XMLNode_t *node);


LIBLAX_EXTERN
XMLNode_t *
XMLNode_convertStringToXMLNode(const char * xml, const XMLNamespaces_t* xmlns);

END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* XMLNode_h */
