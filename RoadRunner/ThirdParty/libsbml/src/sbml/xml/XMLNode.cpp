/**
 * @file    XMLNode.cpp
 * @brief   A node in an XML document tree
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

#include <sstream>

#include <sbml/util/memory.h>
#include <sbml/util/util.h>

/** @cond doxygen-libsbml-internal */
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>
#include <sbml/xml/XMLConstructorException.h>
/** @endcond */

#include <sbml/xml/XMLNode.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

/*
 * @return s with whitespace removed from the beginning and end.
 */
static const string
trim (const string& s)
{
  static const string whitespace(" \t\r\n");

  string::size_type begin = s.find_first_not_of(whitespace);
  string::size_type end   = s.find_last_not_of (whitespace);

  return (begin == string::npos) ? std::string() : s.substr(begin, end - begin + 1);
}


/*
 * Creates a new empty XMLNode with no children.
 */
XMLNode::XMLNode ()
{
}


/*
 * Destroys this XMLNode.
 */
XMLNode::~XMLNode ()
{
}


/*
 * Creates a new XMLNode by copying token.
 */
XMLNode::XMLNode (const XMLToken& token) : XMLToken(token)
{
}


/*
 * Creates a new start element XMLNode with the given set of attributes and
 * namespace declarations.
 */
XMLNode::XMLNode (  const XMLTriple&     triple
                  , const XMLAttributes& attributes
                  , const XMLNamespaces& namespaces
		  , const unsigned int   line
                  , const unsigned int   column) 
                  : XMLToken(triple, attributes, namespaces, line, column)
{
}


/*
 * Creates a start element XMLNode with the given set of attributes.
 */
XMLNode::XMLNode (  const XMLTriple&      triple
                  , const XMLAttributes&  attributes
                  , const unsigned int    line
                  , const unsigned int    column )
                  : XMLToken(triple, attributes, line, column)
{
}  


/*
 * Creates an end element XMLNode with the given set of attributes.
 */
XMLNode::XMLNode (  const XMLTriple&   triple
                  , const unsigned int line
                  , const unsigned int column )
                  : XMLToken(triple, line, column)
{
}


/*
 * Creates a text XMLNode.
 */
XMLNode::XMLNode (  const std::string& chars
                  , const unsigned int line
                  , const unsigned int column )
                  : XMLToken(chars, line, column)
{
}


/** @cond doxygen-libsbml-internal */
/*
 * Creates a new XMLNode by reading XMLTokens from stream.  The stream must
 * be positioned on a start element (stream.peek().isStart() == true) and
 * will be read until the matching end element is found.
 */
XMLNode::XMLNode (XMLInputStream& stream) : XMLToken( stream.next() )
{
  if ( isEnd() ) return;

  std::string s;

  while ( stream.isGood() )
  {
    const XMLToken& next = stream.peek();


    if ( next.isStart() )
    {
      addChild( XMLNode(stream) );
    }
    else if ( next.isText() )
    {
      s = trim(next.getCharacters());
      if (s != "")
        addChild( stream.next() );
      else
        stream.skipText();
    }
    else if ( next.isEnd() )
    {
      stream.next();
      break;
    }
  }
}
/** @endcond */


/*
 * Copy constructor; creates a copy of this XMLNode.
 */
XMLNode::XMLNode(const XMLNode& orig):
      XMLToken (orig)
{
  this->mChildren.assign( orig.mChildren.begin(), orig.mChildren.end() ); 
}


 /*
  * Assignment operator for XMLNode.
  */
XMLNode& 
XMLNode::operator=(const XMLNode& rhs)
{
  if (&rhs == NULL)
  {
    throw XMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    this->XMLToken::operator=(rhs);
    this->mChildren.assign( rhs.mChildren.begin(), rhs.mChildren.end() ); 
  }

  return *this;
}

/*
 * Creates and returns a deep copy of this XMLNode.
 * 
 * @return a (deep) copy of this XMLNode.
 */
XMLNode* 
XMLNode::clone () const
{
  return new XMLNode(*this);
}


/*
 * Adds a copy of child node to this XMLNode.
 */
int
XMLNode::addChild (const XMLNode& node)
{
  /* catch case where node is NULL
   */
  if (&(node) == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }

  if (isStart())
  {
    mChildren.push_back(node);
    /* need to catch the case where this node is both a start and
    * an end element
    */
    if (isEnd()) unsetEnd();
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if (isEOF())
  {
    mChildren.push_back(node);
    // this causes strange things to happen when node is written out
    //   this->mIsStart = true;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_INVALID_XML_OPERATION;
  }

}


/*
 * Inserts a copy of child node as the nth child of this XMLNode.
 */
XMLNode&
XMLNode::insertChild (unsigned int n, const XMLNode& node)
{
  /* catch case where node is NULL
   */
  if (&(node) == NULL)
  {
    return const_cast<XMLNode&>(node);
  }

  unsigned int size = (unsigned int)mChildren.size();

  if ( (n >= size) || (size == 0) )
  {
    mChildren.push_back(node);
    return mChildren.back();
  }

  return *(mChildren.insert(mChildren.begin() + n, node));
}


/*
 * Removes the nth child of this XMLNode and returned the removed node.
 * The caller owns the returned node and is responsible for deleting it.
 *
 * @return the removed child, or NULL if the given index is out of range. 
 */
XMLNode* 
XMLNode::removeChild(unsigned int n)
{
  XMLNode* rval = NULL;

  if ( n < getNumChildren() )
  {
    rval = mChildren[n].clone();
    mChildren.erase(mChildren.begin() + n);
  }
  
  return rval;
}

/* 
 * remove all children
 */
int
XMLNode::removeChildren()
{
  mChildren.clear(); 
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Returns the nth child of this XMLNode.
 */
XMLNode&
XMLNode::getChild (unsigned int n)
{
   return const_cast<XMLNode&>( 
            static_cast<const XMLNode&>(*this).getChild(n)
          );
}


/*
 * Returns the nth child of this XMLNode.
 */
const XMLNode&
XMLNode::getChild (unsigned int n) const
{
  static const XMLNode outOfRange;

  unsigned int size = getNumChildren();
  if ( (n < size) && (size > 0) )
  {
    return mChildren[n];
  }
  else
  {
    // An empty XMLNode object, which is neither start node, 
    // end node, nor text node, returned if the given index 
    // is out of range. 
    // Currently, this object is allocated as a static object
    // to avoid a memory leak.
    // This may be fixed in the futrure release.
    return outOfRange;
  }
}

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
XMLNode&
XMLNode::getChild (const std::string&  name)
{
	return const_cast<XMLNode&>( 
								static_cast<const XMLNode&>(*this).getChild(name)
								);
}
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
const XMLNode& 
XMLNode::getChild (const std::string&  name) const
{
	static const XMLNode outOfRange;
	int index = getIndex(name);
	if (index != -1)
	{
		return getChild((unsigned int)index);
	}
	else 
	{
		// An empty XMLNode object, which is neither start node, 
		// end node, nor text node, returned if the given index 
		// is out of range. 
		// Currently, this object is allocated as a static object
		// to avoid a memory leak.
		// This may be fixed in the futrure release.		
		return outOfRange;
	}

}

/**
 * Return the index of the first child of this XMLNode with the given name.
 *
 *
 * @param name a string, the name of the child for which the 
 * index is required.
 *
 * @return the index of the first child of this XMLNode with the given name, or -1 if not present.
 */
int
XMLNode::getIndex (const std::string& name) const
{
  if (&name == NULL) return -1;
	
  for (unsigned int index = 0; index < getNumChildren(); ++index)
	{
		if (getChild(index).getName() == name) return index;
	}
	
	return -1;
}

/**
 * Compare this XMLNode against another XMLNode returning true if both nodes
 * represent the same XML tree, or false otherwise.
 *
 *
 * @param other another XMLNode to compare against
 *
 * @return boolean indicating whether this XMLNode represents the same XML tree as another.
 */
bool 
XMLNode::equals(const XMLNode& other) const
{
  if (&other == NULL) return false;

	bool equal=true;
	// check if the nodes have the same name,
	equal=(getName()==other.getName());
	// the same namespace uri, 
	equal=(equal && (getURI()==other.getURI()));
	
	XMLAttributes attr1=getAttributes(); 
	XMLAttributes attr2=other.getAttributes();
	int i=0,iMax=attr1.getLength();
	//the same attributes and the same number of children
	equal=(iMax==attr2.getLength());
	std::string attrName;
	while(equal && i<iMax)
	{
		attrName=attr1.getName(i);
		equal=(attr2.getIndex(attrName)!=-1);
		// also check the namspace
		equal=(equal && (attr1.getURI(i)==attr2.getURI(i)));
		++i;
	}
	
	// recursively check all children
	i=0;
	iMax=getNumChildren();
	equal=(equal && (iMax==(int)other.getNumChildren()));
	while(equal && i<iMax)
	{
		equal=getChild(i).equals(other.getChild(i));
		++i;
	}
	return equal; 
}


/**
 * Return a boolean indicating whether this XMLNode has a child with the given name.
 *
 *
 * @param name a string, the name of the child to be checked.
 *
 * @return boolean indicating whether this XMLNode has a child with the given name.
 */
bool 
XMLNode::hasChild (const std::string& name) const
{
	return getIndex(name) != -1;
}

/*
 * @return the number of children for this XMLNode.
 */
unsigned int
XMLNode::getNumChildren () const
{
  return (unsigned int)mChildren.size();
}


/** @cond doxygen-libsbml-internal */
/*
 * Writes this XMLNode and its children to stream.
 */
void
XMLNode::write (XMLOutputStream& stream) const
{
  if (&stream == NULL) return;

  unsigned int children = getNumChildren();

  XMLToken::write(stream);

  if (children > 0)
  {
    bool haveTextNode = false;
    for (unsigned int c = 0; c < children; ++c) 
    {
        const XMLNode& current = getChild(c);
        stream << current;
        haveTextNode |= current.isText();
    }

    if (!mTriple.isEmpty())
    {
      // edge case ... we have an element with a couple of elements, and 
      // one is a text node (ugly!) in this case we can get a hanging
      // indent ... so we downindent ... 
      if (children > 1 && haveTextNode)
      {
        stream.downIndent();
      }
      stream.endElement( mTriple );
    }
  }
  else if ( isStart() && !isEnd() ) 
  {
    stream.endElement( mTriple );
  }

}
/** @endcond */


/*
 * Returns a string which is converted from this XMLNode.
 */
std::string XMLNode::toXMLString() const
{
  std::ostringstream oss;
  XMLOutputStream xos(oss,"UTF-8",false);
  write(xos);

  return oss.str();
}


/*
 * Returns a string which is converted from a given XMLNode.
 */
std::string XMLNode::convertXMLNodeToString(const XMLNode* xnode)
{
  if(xnode == NULL) return "";

  std::ostringstream oss;
  XMLOutputStream xos(oss,"UTF-8",false);
  xnode->write(xos);

  return oss.str();
}


/*
 * Returns a XMLNode which is converted from a given string.
 */
XMLNode* XMLNode::convertStringToXMLNode(const std::string& xmlstr, const XMLNamespaces* xmlns)
{
  if (&xmlstr == NULL) return NULL;

  XMLNode* xmlnode     = NULL;
  std::ostringstream oss;
  const char* dummy_xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
  const char* dummy_element_start = "<dummy";
  const char* dummy_element_end   = "</dummy>";


  oss << dummy_xml;
  oss << dummy_element_start;
  if(xmlns != NULL)
  {
    for(int i=0; i < xmlns->getLength(); i++)
    {
      oss << " xmlns";
      if(xmlns->getPrefix(i) != "") oss << ":" << xmlns->getPrefix(i);
      oss << "=\"" << xmlns->getURI(i) << '"';
    }
  }
  oss << ">";
  oss << xmlstr;
  oss << dummy_element_end;


  const char* xmlstr_c = safe_strdup(oss.str().c_str());
  XMLInputStream xis(xmlstr_c,false);
  XMLNode* xmlnode_tmp = new XMLNode(xis);

  if(xis.isError() || (xmlnode_tmp->getNumChildren() == 0) )
  {
    delete xmlnode_tmp;
    return NULL;
  }


  /**
   * this is fine if the first child is a parent element
   * it actually falls down if all your elements have equal footing
   * eg 
   *  <p>The following is MathML markup:</p>
   *  <p xmlns="http://www.w3.org/1999/xhtml"> Test2 </p>
   */

  if (xmlnode_tmp->getNumChildren() == 1)
  {
    xmlnode = new XMLNode(xmlnode_tmp->getChild(0));
  }
  else
  {
    xmlnode = new XMLNode();
    for(unsigned int i=0; i < xmlnode_tmp->getNumChildren(); i++)
    {
      xmlnode->addChild(xmlnode_tmp->getChild(i));
    }
  }

  delete xmlnode_tmp;
  safe_free(const_cast<char*>(xmlstr_c));

  return xmlnode;
}


/** @cond doxygen-libsbml-internal */
/*
 * Inserts this XMLNode and its children into stream.
 */
LIBLAX_EXTERN
XMLOutputStream& operator<< (XMLOutputStream& stream, const XMLNode& node)
{
  node.write(stream);
  return stream;
}
/** @endcond */


/** @cond doxygen-c-only */


/**
 * Creates a new empty XMLNode_t structure with no children
 * and returns a pointer to it.
 *
 * @return pointer to the new XMLNode_t structure.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_create (void)
{
  return new(nothrow) XMLNode;
}


/**
 * Creates a new XMLNode_t structure by copying token and returns a pointer
 * to it.
 *
 * @param token XMLToken_t structure to be copied to XMLNode_t structure.
 *
 * @return pointer to the new XMLNode_t structure.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_createFromToken (const XMLToken_t *token)
{
  if (token == NULL) return NULL;
  return new(nothrow) XMLNode(*token);
}


/**
 * Creates a new start element XMLNode_t structure with XMLTriple_t, 
 * XMLAttributes_t and XMLNamespaces_t structures set and returns a 
 * pointer to it.
 *
 * @param triple XMLTriple_t structure to be set.
 * @param attr XMLAttributes_t structure to be set.
 * @param ns XMLNamespaces_t structure to be set.
 *
 * @return pointer to new XMLNode_t structure.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_createStartElementNS (const XMLTriple_t     *triple,
			      const XMLAttributes_t *attr,
			      const XMLNamespaces_t *ns)
{
  if (triple == NULL || attr == NULL || ns == NULL) return NULL;
  return new(nothrow) XMLNode(*triple, *attr, *ns);
}


/**
 * Creates a new start element XMLNode_t structure with XMLTriple_t 
 * and XMLAttributes_t structures set and returns a pointer to it.
 *
 * @param triple XMLTriple_t structure to be set.
 * @param attr XMLAttributes_t structure to be set.
 *
 * @return pointer to new XMLNode_t structure.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_createStartElement  (const XMLTriple_t *triple,
			     const XMLAttributes_t *attr)
{
  if (triple == NULL || attr == NULL) return NULL;
  return new(nothrow) XMLNode(*triple, *attr);
}


/**
 * Creates a new end element XMLNode_t structure with XMLTriple_t 
 * structure set and returns a pointer to it.
 *
 * @param triple XMLTriple_t structure to be set.
 *
 * @return pointer to new XMLNode_t structure.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_createEndElement (const XMLTriple_t *triple)
{
  if (triple == NULL) return NULL;
  return new(nothrow) XMLNode(*triple);
}


LIBLAX_EXTERN
XMLNode_t *
XMLNode_createTextNode (const char *text)
{
  return (text != NULL) ? new(nothrow) XMLNode(text) : new(nothrow) XMLNode;
}


#if 0

/**
 * Creates a new XMLNode_t structure by reading XMLTokens from stream.  
 *
 * The stream must
 * be positioned on a start element (stream.peek().isStart() == true) and
 * will be read until the matching end element is found.
 *
 * @param stream XMLInputStream from which XMLNode_t structure is to be created.
 *
 * @return pointer to the new XMLNode_t structure.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_createFromStream (XMLInputStream_t *stream)
{
  return new(nothrow) XMLNode(stream);
}

#endif

/**
 * Creates a deep copy of the given XMLNode_t structure
 * 
 * @param n the XMLNode_t structure to be copied
 * 
 * @return a (deep) copy of the given XMLNode_t structure.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_clone (const XMLNode_t* n)
{
  if (n == NULL) return NULL;
  return static_cast<XMLNode*>( n->clone() );
}


/**
 * Destroys this XMLNode_t structure.
 *
 * @param node XMLNode_t structure to be freed.
 */
LIBLAX_EXTERN
void
XMLNode_free (XMLNode_t *node)
{
  if (node == NULL) return;
  delete static_cast<XMLNode*>(node);
}


/**
 * Adds a copy of child node to this XMLNode_t structure.
 *
 * @param node XMLNode_t structure to which child is to be added.
 * @param child XMLNode_t structure to be added as child.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int
XMLNode_addChild (XMLNode_t *node, const XMLNode_t *child)
{
  if (node == NULL || child == NULL) return LIBSBML_INVALID_OBJECT;
  return node->addChild(*child);
}


/**
 * Inserts a copy of child node to this XMLNode_t structure.
 *
 * @param node XMLNode_t structure to which child is to be added.
 * @pram n the index at which the given node is inserted
 * @param child XMLNode_t structure to be inserted as nth child.
 *
 * @return the newly inserted child in this XMLNode. 
 * NULL will be returned if the given child is NULL. 
 */
LIBLAX_EXTERN
XMLNode_t*
XMLNode_insertChild (XMLNode_t *node, unsigned int n, const XMLNode_t *child)
{
  if (node == NULL || child == NULL )
  {
    return NULL;
  }

  return &(node->insertChild(n, *child));
}


/**
 * Removes the nth child of this XMLNode and returned the removed node.
 *
 * @param node XMLNode_t structure to which child is to be removed.
 * @param n the index of the node to be removed
 *
 * @return the removed child, or NULL if the given index is out of range. 
 *
 * @note This function invalidates all existing references to child nodes 
 * after the position or first.
 */
LIBLAX_EXTERN
XMLNode_t* 
XMLNode_removeChild(XMLNode_t *node, unsigned int n)
{
  if (node == NULL) return NULL;
  return node->removeChild(n);
}


/**
 * Removes all children from this node.
 *
 * @param n an integer the index of the resource to be deleted
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int
XMLNode_removeChildren (XMLNode_t *node)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->removeChildren();
}


/**
 * Returns the text of this element.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the characters of this XML text.
 */
LIBLAX_EXTERN
const char *
XMLNode_getCharacters (const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return node->getCharacters().empty() ? NULL : node->getCharacters().c_str();
}


/**
 * Sets the XMLTripe (name, uri and prefix) of this XML element.
 * Nothing will be done if this XML element is a text node.
 *
 * @param node XMLNode_t structure to which the triple to be added.
 * @param triple an XMLTriple, the XML triple to be set to this XML element.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_setTriple(XMLNode_t *node, const XMLTriple_t *triple)
{
  if(node == NULL || triple == NULL) return LIBSBML_INVALID_OBJECT;
  return node->setTriple(*triple);
}


/**
 * Returns the (unqualified) name of this XML element.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the (unqualified) name of this XML element.
 */
LIBLAX_EXTERN
const char *
XMLNode_getName (const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return node->getName().empty() ? NULL : node->getName().c_str();
}


/**
 * Returns the namespace prefix of this XML element.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the namespace prefix of this XML element.  
 *
 * @note If no prefix
 * exists, an empty string will be return.
 */
LIBLAX_EXTERN
const char *
XMLNode_getPrefix (const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return node->getPrefix().empty() ? NULL : node->getPrefix().c_str();
}


/**
 * Returns the namespace URI of this XML element.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the namespace URI of this XML element.
 */
LIBLAX_EXTERN
const char *
XMLNode_getURI (const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return node->getURI().empty() ? NULL : node->getURI().c_str();
}


/**
 * Returns the nth child of this XMLNode_t structure.
 *
 * @param node XMLNode_t structure to be queried.
 * @param n the index of the node to return
 *
 * @return the nth child of this XMLNode_t structure.
 */
LIBLAX_EXTERN
const XMLNode_t *
XMLNode_getChild (const XMLNode_t *node, const int n)
{
  if (node == NULL) return NULL;
  return &(node->getChild(n));
}


/**
 * Returns the (non-const) nth child of this XMLNode_t structure.
 *
 * @param node XMLNode_t structure to be queried.
 * @param n the index of the node to return
 *
 * @return the non-const nth child of this XMLNode_t structure.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_getChildNC (XMLNode_t *node, const unsigned int n)
{
  if (node == NULL) return NULL;
  return &(node->getChild(n));
}

/**
 * Returns the (non-const) the first child of the XMLNode_t structure node with the given name.
 *
 * If no child with corrsponding name can be found, 
 * this method returns an empty node.
 *
 * @param node XMLNode_t structure to be queried.
 * @param name the name of the node to return
 * 
 * @return the first child of this XMLNode with given name.
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_getChildForNameNC (XMLNode_t *node, const char*  name)
{
  if (node == NULL) return NULL;
	return &(node->getChild(name));
}

/**
 * Returns the first child of the XMLNode_t structure node with the given name.
 *
 * If no child with corrsponding name can be found, 
 * this method returns an empty node.
 *
 * @param node XMLNode_t structure to be queried.
 * @param name the name of the node to return
 * 
 * @return the first child of this XMLNode with given name.
 */
LIBLAX_EXTERN
const XMLNode_t *
XMLNode_getChildForName (const XMLNode_t *node, const char*  name)
{
  if (node == NULL) return NULL;
	return &(node->getChild(name));
}

/**
 * Return the index of the first child of the XMLNode_t structure node with the given name.
 *
 * @param node XMLNode_t structure to be queried.
 * @param name a string, the name of the child for which the 
 * index is required.
 *
 * @return the index of the first child of node with the given name, or -1 if not present.
 */
LIBLAX_EXTERN
int 
XMLNode_getIndex (const XMLNode_t *node, const char*  name)
{
  if (node == NULL) return -1;
	return (node->getIndex(name));
}

/**
 * Return a boolean indicating whether node has a child with the given name.
 *
 * @param node XMLNode_t structure to be queried.
 * @param name a string, the name of the child to be checked.
 *
 * @return true (non-zero) if this node has a child with the given name false (zero) otherwise.
 */
LIBLAX_EXTERN
int 
XMLNode_hasChild (const XMLNode_t *node, const char*  name)
{
  if (node == NULL) return (int)false;
	return static_cast<int>( node->hasChild(name) );
}

/**
 * Compare one XMLNode against another XMLNode returning true (non-zero) if both nodes
 * represent the same XML tree, or false (zero) otherwise.
 *
 *
 * @param other another XMLNode to compare against
 *
 * @return true (non-zero) if both nodes
 * represent the same XML tree, or false (zero) otherwise
 */
LIBLAX_EXTERN
int 
XMLNode_equals(const XMLNode_t *node, const XMLNode_t* other)
{
  if (node == NULL && other == NULL) return (int)true;
  if (node == NULL || other == NULL) return (int)false;
	return static_cast<int>( node->equals(*other) );
}

/**
 * Returns the number of children for this XMLNode_t structure.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the number of children for this XMLNode_t structure.
 */
LIBLAX_EXTERN
unsigned int
XMLNode_getNumChildren (const XMLNode_t *node)
{
  if (node == NULL) return 0;
  return node->getNumChildren();
}



/**
 * Returns the attributes of this element.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the XMLAttributes_t of this XML element.
 */
LIBLAX_EXTERN
const XMLAttributes_t *
XMLNode_getAttributes (const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return &(node->getAttributes());
}


/**
 * Sets an XMLAttributes to this XMLNode.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure to which attributes to be set.
 * @param attributes XMLAttributes to be set to this XMLNode.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note This function replaces the existing XMLAttributes with the new one.
 */
LIBLAX_EXTERN
int 
XMLNode_setAttributes(XMLNode_t *node, const XMLAttributes_t* attributes)
{
  if (node == NULL || attributes == NULL) return LIBSBML_INVALID_OBJECT;
  return node->setAttributes(*attributes);
}


/**
 * Adds an attribute with the given local name to the attribute set in this XMLNode.
 * (namespace URI and prefix are empty)
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure to which an attribute to be added.
 * @param name a string, the local name of the attribute.
 * @param value a string, the value of the attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note if the local name without namespace URI already exists in the
 * attribute set, its value will be replaced.
 *
 */
LIBLAX_EXTERN
int 
XMLNode_addAttr ( XMLNode_t *node,  const char* name, const char* value )
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->addAttr(name, value, "", "");
}


/**
 * Adds an attribute with a prefix and namespace URI to the attribute set 
 * in this XMLNode optionally 
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure to which an attribute to be added.
 * @param name a string, the local name of the attribute.
 * @param value a string, the value of the attribute.
 * @param namespaceURI a string, the namespace URI of the attribute.
 * @param prefix a string, the prefix of the namespace
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note if local name with the same namespace URI already exists in the
 * attribute set, its value and prefix will be replaced.
 *
 */
LIBLAX_EXTERN
int 
XMLNode_addAttrWithNS ( XMLNode_t *node,  const char* name
	                , const char* value
    	                , const char* namespaceURI
	                , const char* prefix      )
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->addAttr(name, value, namespaceURI, prefix);
}



/**
 * Adds an attribute with the given XMLTriple/value pair to the attribute set
 * in this XMLNode.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @note if local name with the same namespace URI already exists in the 
 * attribute set, its value and prefix will be replaced.
 *
 * @param node XMLNode_t structure to which an attribute to be added.
 * @param triple an XMLTriple, the XML triple of the attribute.
 * @param value a string, the value of the attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_addAttrWithTriple (XMLNode_t *node, const XMLTriple_t *triple, const char* value)
{
  if (node == NULL || triple == NULL) return LIBSBML_INVALID_OBJECT;
  return node->addAttr(*triple, value);
}


/**
 * Removes an attribute with the given index from the attribute set in
 * this XMLNode.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure from which an attribute to be removed.
 * @param n an integer the index of the resource to be deleted
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_removeAttr (XMLNode_t *node, int n)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->removeAttr(n);
}


/**
 * Removes an attribute with the given local name (without namespace URI) 
 * from the attribute set in this XMLNode.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure from which an attribute to be removed.
 * @param name   a string, the local name of the attribute.
 * @param uri    a string, the namespace URI of the attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_removeAttrByName (XMLNode_t *node, const char* name)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->removeAttr(name, "");
}


/**
 * Removes an attribute with the given local name and namespace URI from 
 * the attribute set in this XMLNode.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure from which an attribute to be removed.
 * @param name   a string, the local name of the attribute.
 * @param uri    a string, the namespace URI of the attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_removeAttrByNS (XMLNode_t *node, const char* name, const char* uri)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->removeAttr(name, uri);
}


/**
 * Removes an attribute with the given XMLTriple from the attribute set 
 * in this XMLNode.  
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure from which an attribute to be removed.
 * @param triple an XMLTriple, the XML triple of the attribute.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_removeAttrByTriple (XMLNode_t *node, const XMLTriple_t *triple)
{
  if (node == NULL || triple == NULL) return LIBSBML_INVALID_OBJECT;
  return node->removeAttr(*triple);
}


/**
 * Clears (deletes) all attributes in this XMLNode.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure from which attributes to be cleared.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_clearAttributes(XMLNode_t *node)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->clearAttributes();
}



/**
 * Return the index of an attribute with the given local name and namespace URI.
 *
 * @param node XMLNode_t structure to be queried.
 * @param name a string, the local name of the attribute.
 * @param uri  a string, the namespace URI of the attribute.
 *
 * @return the index of an attribute with the given local name and namespace URI, 
 * or -1 if not present.
 *
 */
LIBLAX_EXTERN
int 
XMLNode_getAttrIndex (const XMLNode_t *node, const char* name, const char* uri)
{
  if (node == NULL) return -1;
  return node->getAttrIndex(name, uri);
}


/**
 * Return the index of an attribute with the given XMLTriple.
 *
 * @param node XMLNode_t structure to be queried.
 * @param triple an XMLTriple, the XML triple of the attribute for which 
 *        the index is required.
 *
 * @return the index of an attribute with the given XMLTriple, or -1 if not present.
 */
LIBLAX_EXTERN
int 
XMLNode_getAttrIndexByTriple (const XMLNode_t *node, const XMLTriple_t *triple)
{
  if (node == NULL || triple == NULL) return -1;
  return node->getAttrIndex(*triple);
}


/**
 * Return the number of attributes in the attributes set.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the number of attributes in the attributes set in this XMLNode.
 */
LIBLAX_EXTERN
int 
XMLNode_getAttributesLength (const XMLNode_t *node)
{
  if (node == NULL) return 0;
  return node->getAttributesLength();
}


/**
 * Return the local name of an attribute in the attributes set in this 
 * XMLNode (by position).
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, the position of the attribute whose local name 
 * is required.
 *
 * @return the local name of an attribute in this list (by position).  
 *
 * @note If index
 * is out of range, an empty string will be returned.  Use XMLNode_hasAttr(...) 
 * to test for the attribute existence.
 */
LIBLAX_EXTERN
char* 
XMLNode_getAttrName (const XMLNode_t *node, int index)
{
  if (node == NULL) return NULL;
  
  const std::string str = node->getAttrName(index);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Return the prefix of an attribute in the attribute set in this 
 * XMLNode (by position).
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, the position of the attribute whose prefix is 
 * required.
 *
 * @return the namespace prefix of an attribute in the attribute set
 * (by position).  
 *
 * @note If index is out of range, an empty string will be
 * returned. Use XMLNode_hasAttr(...) to test for the attribute existence.
 */
LIBLAX_EXTERN
char* 
XMLNode_getAttrPrefix (const XMLNode_t *node, int index)
{
  if (node == NULL) return NULL;

  const std::string str = node->getAttrPrefix(index);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Return the prefixed name of an attribute in the attribute set in this 
 * XMLNode (by position).
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, the position of the attribute whose prefixed 
 * name is required.
 *
 * @return the prefixed name of an attribute in the attribute set 
 * (by position).  
 *
 * @note If index is out of range, an empty string will be
 * returned.  Use XMLNode_hasAttr(...) to test for attribute existence.
 */
LIBLAX_EXTERN
char* 
XMLNode_getAttrPrefixedName (const XMLNode_t *node, int index)
{
  if (node == NULL) return NULL;

  const std::string str = node->getAttrPrefixedName(index);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Return the namespace URI of an attribute in the attribute set in this 
 * XMLNode (by position).
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, the position of the attribute whose namespace 
 * URI is required.
 *
 * @return the namespace URI of an attribute in the attribute set (by position).
 *
 * @note If index is out of range, an empty string will be returned.  Use
 * XMLNode_hasAttr(index) to test for attribute existence.
 */
LIBLAX_EXTERN
char* 
XMLNode_getAttrURI (const XMLNode_t *node, int index)
{
  if (node == NULL) return NULL;
  const std::string str = node->getAttrURI(index);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Return the value of an attribute in the attribute set in this XMLNode  
 * (by position).
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, the position of the attribute whose value is 
 * required.
 *
 * @return the value of an attribute in the attribute set (by position).  
 *
 * @note If index
 * is out of range, an empty string will be returned. Use XMLNode_hasAttr(...)
 * to test for attribute existence.
 */
LIBLAX_EXTERN
char* 
XMLNode_getAttrValue (const XMLNode_t *node, int index)
{
  if (node == NULL) return NULL;

  const std::string str = node->getAttrValue(index);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}



/**
 * Return a value of an attribute with the given local name (without namespace URI).
 *
 * @param node XMLNode_t structure to be queried.
 * @param name a string, the local name of the attribute whose value is required.
 *
 * @return The attribute value as a string.  
 *
 * @note If an attribute with the given local name (without namespace URI) 
 * does not exist, an empty string will be returned.  
 * Use XMLNode_hasAttr(...) to test for attribute existence.
 */
LIBLAX_EXTERN
char* 
XMLNode_getAttrValueByName (const XMLNode_t *node, const char* name)
{
  if (node == NULL) return NULL;

  const std::string str = node->getAttrValue(name, "");

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Return a value of an attribute with the given local name and namespace URI.
 *
 * @param node XMLNode_t structure to be queried.
 * @param name a string, the local name of the attribute whose value is required.
 * @param uri  a string, the namespace URI of the attribute.
 *
 * @return The attribute value as a string.  
 *
 * @note If an attribute with the 
 * given local name and namespace URI does not exist, an empty string will be 
 * returned.  
 * Use XMLNode_hasAttr(name, uri) to test for attribute existence.
 */
LIBLAX_EXTERN
char* 
XMLNode_getAttrValueByNS (const XMLNode_t *node, const char* name, const char* uri)
{
  if (node == NULL) return NULL;
  const std::string str = node->getAttrValue(name, uri);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Return a value of an attribute with the given XMLTriple.
 *
 * @param node XMLNode_t structure to be queried.
 * @param triple an XMLTriple, the XML triple of the attribute whose 
 *        value is required.
 *
 * @return The attribute value as a string.  
 *
 * @note If an attribute with the
 * given XMLTriple does not exist, an empty string will be returned.  
 * Use XMLNode_hasAttr(...) to test for attribute existence.
 */
LIBLAX_EXTERN
char* 
XMLNode_getAttrValueByTriple (const XMLNode_t *node, const XMLTriple_t *triple)
{
  if (node == NULL || triple == NULL) return NULL;
  const std::string str = node->getAttrValue(*triple);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Predicate returning @c true or @c false depending on whether
 * an attribute with the given index exists in the attribute set in this 
 * XMLNode.
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, the position of the attribute.
 *
 * @return @c non-zero (true) if an attribute with the given index exists in 
 * the attribute set in this XMLNode, @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_hasAttr (const XMLNode_t *node, int index)
{
  if (node == NULL) return (int)false;
  return node->hasAttr(index);
}


/**
 * Predicate returning @c true or @c false depending on whether
 * an attribute with the given local name (without namespace URI) 
 * exists in the attribute set in this XMLNode.
 *
 * @param node XMLNode_t structure to be queried.
 * @param name a string, the local name of the attribute.
 *
 * @return @c non-zero (true) if an attribute with the given local name 
 * (without namespace URI) exists in the attribute set in this XMLNode, 
 * @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_hasAttrWithName (const XMLNode_t *node, const char* name)
{
  if (node == NULL) return (int)false;
  return node->hasAttr(name, "");
}


/**
 * Predicate returning @c true or @c false depending on whether
 * an attribute with the given local name and namespace URI exists 
 * in the attribute set in this XMLNode.
 *
 * @param node XMLNode_t structure to be queried.
 * @param name a string, the local name of the attribute.
 * @param uri  a string, the namespace URI of the attribute.
 *
 * @return @c non-zero (true) if an attribute with the given local name 
 * and namespace URI exists in the attribute set in this XMLNode, 
 * @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_hasAttrWithNS (const XMLNode_t *node, const char* name, const char* uri)
{
  if (node == NULL) return (int)false;
  return node->hasAttr(name, uri);
}


/**
 * Predicate returning @c true or @c false depending on whether
 * an attribute with the given XML triple exists in the attribute set in 
 * this XMLNode 
 *
 * @param node XMLNode_t structure to be queried.
 * @param triple an XMLTriple, the XML triple of the attribute 
 *
 * @return @c non-zero (true) if an attribute with the given XML triple exists
 * in the attribute set in this XMLNode, @c zero (false) otherwise.
 *
 */
LIBLAX_EXTERN
int
XMLNode_hasAttrWithTriple (const XMLNode_t *node, const XMLTriple_t *triple)
{
  if (node == NULL || triple == NULL) return (int)false;
  return node->hasAttr(*triple);
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * the attribute set in this XMLNode set is empty.
 * 
 * @param node XMLNode_t structure to be queried.
 *
 * @return @c non-zero (true) if the attribute set in this XMLNode is empty, 
 * @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_isAttributesEmpty (const XMLNode_t *node)
{
  if (node == NULL) return (int)false;
  return node->isAttributesEmpty();
}



/**
 * Returns the XML namespace declarations for this XML element.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the XML namespace declarations for this XML element.
 */
LIBLAX_EXTERN
const XMLNamespaces_t *
XMLNode_getNamespaces (const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return &(node->getNamespaces());
}


/**
 * Sets an XMLnamespaces to this XML element.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure to be queried.
 * @param namespaces XMLNamespaces to be set to this XMLNode.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note This function replaces the existing XMLNamespaces with the new one.
 */
LIBLAX_EXTERN
int 
XMLNode_setNamespaces(XMLNode_t *node, const XMLNamespaces_t* namespaces)
{
  if (node == NULL || namespaces == NULL) return LIBSBML_INVALID_OBJECT;
  return node->setNamespaces(*namespaces);
}


/**
 * Appends an XML namespace prefix and URI pair to this XMLNode.
 * If there is an XML namespace with the given prefix in this XMLNode, 
 * then the existing XML namespace will be overwritten by the new one.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure to be queried.
 * @param uri a string, the uri for the namespace
 * @param prefix a string, the prefix for the namespace
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_addNamespace (XMLNode_t *node, const char* uri, const char* prefix)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->addNamespace(uri, prefix);
}


/**
 * Removes an XML Namespace stored in the given position of the XMLNamespaces
 * of this XMLNode.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, position of the removed namespace.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_removeNamespace (XMLNode_t *node, int index)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->removeNamespace(index);
}


/**
 * Removes an XML Namespace with the given prefix.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure to be queried.
 * @param prefix a string, prefix of the required namespace.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_removeNamespaceByPrefix (XMLNode_t *node, const char* prefix)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->removeNamespace(prefix);
}


/**
 * Clears (deletes) all XML namespace declarations in the XMLNamespaces 
 * of this XMLNode.
 * Nothing will be done if this XMLNode is not a start element.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_XML_OPERATION
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int 
XMLNode_clearNamespaces (XMLNode_t *node)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->clearNamespaces();
}


/**
 * Look up the index of an XML namespace declaration by URI.
 *
 * @param node XMLNode_t structure to be queried.
 * @param uri a string, uri of the required namespace.
 *
 * @return the index of the given declaration, or -1 if not present.
 */
LIBLAX_EXTERN
int 
XMLNode_getNamespaceIndex (const XMLNode_t *node, const char* uri)
{
  if (node == NULL) return -1;
  return node->getNamespaceIndex(uri);
}


/**
 * Look up the index of an XML namespace declaration by prefix.
 *
 * @param node XMLNode_t structure to be queried.
 * @param prefix a string, prefix of the required namespace.
 *
 * @return the index of the given declaration, or -1 if not present.
 */
LIBLAX_EXTERN
int 
XMLNode_getNamespaceIndexByPrefix (const XMLNode_t *node, const char* prefix)
{
  if (node == NULL) return -1;
  return node->getNamespaceIndexByPrefix(prefix);
}


/**
 * Returns the number of XML namespaces stored in the XMLNamespaces 
 * of this XMLNode.
 *
 * @param node XMLNode_t structure to be queried.
 *
 * @return the number of namespaces in this list.
 */
LIBLAX_EXTERN
int 
XMLNode_getNamespacesLength (const XMLNode_t *node)
{
  if (node == NULL) return 0;
  return node->getNamespacesLength();
}


/**
 * Look up the prefix of an XML namespace declaration by position.
 *
 * Callers should use getNamespacesLength() to find out how many 
 * namespaces are stored in the XMLNamespaces.
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, position of the removed namespace.
 * 
 * @return the prefix of an XML namespace declaration in the XMLNamespaces 
 * (by position).  
 *
 * @note returned char* should be freed with safe_free() by the caller.
 */
LIBLAX_EXTERN
char* 
XMLNode_getNamespacePrefix (const XMLNode_t *node, int index)
{
  if (node == NULL) return NULL;
  const std::string str = node->getNamespacePrefix(index);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Look up the prefix of an XML namespace declaration by its URI.
 *
 * @param node XMLNode_t structure to be queried.
 * @param uri a string, uri of the required namespace.
 *
 * @return the prefix of an XML namespace declaration given its URI.  
 *
 * @note returned char* should be freed with safe_free() by the caller.
 */
LIBLAX_EXTERN
char* 
XMLNode_getNamespacePrefixByURI (const XMLNode_t *node, const char* uri)
{
  if (node == NULL) return NULL;
  const std::string str = node->getNamespacePrefix(uri);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Look up the URI of an XML namespace declaration by its position.
 *
 * @param node XMLNode_t structure to be queried.
 * @param index an integer, position of the removed namespace.
 *
 * @return the URI of an XML namespace declaration in the XMLNamespaces
 * (by position).  
 *
 * @note returned char* should be freed with safe_free() by the caller.
 */
LIBLAX_EXTERN
char* 
XMLNode_getNamespaceURI (const XMLNode_t *node, int index)
{
  if (node == NULL) return NULL;
  const std::string str = node->getNamespaceURI(index);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Look up the URI of an XML namespace declaration by its prefix.
 *
 * @param node XMLNode_t structure to be queried.
 * @param prefix a string, prefix of the required namespace.
 *
 * @return the URI of an XML namespace declaration given its prefix.  
 *
 * @note returned char* should be freed with safe_free() by the caller.
 */
LIBLAX_EXTERN
char* 
XMLNode_getNamespaceURIByPrefix (const XMLNode_t *node, const char* prefix)
{
  if (node == NULL) return NULL;
  const std::string str = node->getNamespaceURI(prefix);

  return str.empty() ? NULL : safe_strdup(str.c_str());
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * the XMLNamespaces of this XMLNode is empty.
 * 
 * @param node XMLNode_t structure to be queried.
 *
 * @return @c non-zero (true) if the XMLNamespaces of this XMLNode is empty, 
 * @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_isNamespacesEmpty (const XMLNode_t *node)
{
  if (node == NULL) return (int)false;
  return node->isNamespacesEmpty();
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * an XML Namespace with the given URI is contained in the XMLNamespaces of
 * this XMLNode.
 * 
 * @param node XMLNode_t structure to be queried.
 * @param uri a string, the uri for the namespace
 *
 * @return @c no-zero (true) if an XML Namespace with the given URI is 
 * contained in the XMLNamespaces of this XMLNode,  @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_hasNamespaceURI(const XMLNode_t *node, const char* uri)
{
  if (node == NULL) return (int) false;
  return node->hasNamespaceURI(uri);
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * an XML Namespace with the given prefix is contained in the XMLNamespaces of
 * this XMLNode.
 *
 * @param node XMLNode_t structure to be queried.
 * @param prefix a string, the prefix for the namespace
 * 
 * @return @c no-zero (true) if an XML Namespace with the given URI is 
 * contained in the XMLNamespaces of this XMLNode, @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_hasNamespacePrefix(const XMLNode_t *node, const char* prefix)
{
  if (node == NULL) return (int)false;
  return node->hasNamespacePrefix(prefix);
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * an XML Namespace with the given uri/prefix pair is contained in the 
 * XMLNamespaces of this XMLNode.
 *
 * @param node XMLNode_t structure to be queried.
 * @param uri a string, the uri for the namespace
 * @param prefix a string, the prefix for the namespace
 * 
 * @return @c non-zero (true) if an XML Namespace with the given uri/prefix pair is 
 * contained in the XMLNamespaces of this XMLNode,  @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_hasNamespaceNS(const XMLNode_t *node, const char* uri, const char* prefix)
{
  if (node == NULL) return (int)false;
  return node->hasNamespaceNS(uri, prefix);
}



/**
 * Returns a string which is converted from a given XMLNode. 
 *
 * @param node XMLNode_t to be converted to a string.
 *
 * @return a string (char*) which is converted from a given XMLNode.
 *
 * @note returned char* should be freed with safe_free() by the caller.
 */
LIBLAX_EXTERN
char *
XMLNode_toXMLString(const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return safe_strdup(node->toXMLString().c_str());
}


/**
 * Returns a string which is converted from a given XMLNode. 
 *
 * @param node XMLNode_t to be converted to a string.
 *
 * @return a string (char*) which is converted from a given XMLNode.
 *
 * @note returned char* should be freed with safe_free() by the caller.
 */
LIBLAX_EXTERN
const char *
XMLNode_convertXMLNodeToString(const XMLNode_t *node)
{
  if (node == NULL) return NULL;
  return safe_strdup((XMLNode::convertXMLNodeToString(node)).c_str());
}


/**
 * Returns an XMLNode_t pointer which is converted from a given string containing
 * XML content.
 *
 * XMLNamespaces (the second argument) must be given if the corresponding 
 * xmlns attribute is not included in the string of the first argument. 
 *
 * @param xml string to be converted to a XML node.
 * @param xmlns XMLNamespaces_t structure the namespaces to set.
 *
 * @return pointer to XMLNode_t structure which is converted from a given string. 
 */
LIBLAX_EXTERN
XMLNode_t *
XMLNode_convertStringToXMLNode(const char * xml, const XMLNamespaces_t* xmlns)
{
  if (xml == NULL) return NULL;
  return XMLNode::convertStringToXMLNode(xml, xmlns);
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * this XMLNode_t structure is an XML element.
 * 
 * @param node XMLNode_t structure to be queried.
 *
 * @return @c non-zero (true) if this XMLNode_t structure is an XML element, @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_isElement (const XMLNode_t *node)
{
  if (node == NULL ) return (int)false;
  return static_cast<int>( node->isElement() );
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * this XMLNode_t structure is an XML end element.
 * 
 * @param node XMLNode_t structure to be queried.
 *
 * @return @c non-zero (true) if this XMLNode_t structure is an XML end element, @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_isEnd (const XMLNode_t *node) 
{
  if (node == NULL) return (int)false;
  return static_cast<int>( node->isEnd() );
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * this XMLNode_t structure is an XML end element for the given start element.
 * 
 * @param node XMLNode_t structure to be queried.
 * @param element XMLNode_t structure, element for which query is made.
 *
 * @return @c non-zero (true) if this XMLNode_t structure is an XML end element for the given
 * XMLNode_t structure start element, @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_isEndFor (const XMLNode_t *node, const XMLNode_t *element)
{
  if (node == NULL) return (int)false;
  return static_cast<int>( node->isEndFor(*element) );
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * this XMLNode_t structure is an end of file marker.
 * 
 * @param node XMLNode_t structure to be queried.
 *
 * @return @c non-zero (true) if this XMLNode_t structure is an end of file (input) marker, @c zero (false)
 * otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_isEOF (const XMLNode_t *node)
{
  if (node == NULL) return (int) false;
  return static_cast<int>( node->isEOF() );
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * this XMLNode_t structure is an XML start element.
 * 
 * @param node XMLNode_t structure to be queried.
 *
 * @return @c true if this XMLNode_t structure is an XML start element, @c false otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_isStart (const XMLNode_t *node)
{
  if (node == NULL) return (int)false;
  return static_cast<int>( node->isStart() );
}


/**
 * Predicate returning @c true or @c false depending on whether 
 * this XMLNode_t structure is an XML text element.
 * 
 * @param node XMLNode_t structure to be queried.
 *
 * @return @c non-zero (true) if this XMLNode_t structure is an XML text element, @c zero (false) otherwise.
 */
LIBLAX_EXTERN
int
XMLNode_isText (const XMLNode_t *node)
{
  if (node == NULL) return (int)false;
  return static_cast<int>( node->isText() );
}


/**
 * Declares this XML start element is also an end element.
 *
 * @param node XMLNode_t structure to be set.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int
XMLNode_setEnd (XMLNode_t *node)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->setEnd();
}


/**
 * Declares this XMLNode_t structure is an end-of-file (input) marker.
 *
 * @param node XMLNode_t structure to be set.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int
XMLNode_setEOF (XMLNode_t *node)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->setEOF();
}


/*
 * Declares this XML start/end element is no longer an end element.
 *
 * @param node XMLNode_t structure to be set.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBLAX_EXTERN
int
XMLNode_unsetEnd (XMLNode_t *node)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->unsetEnd();
}


/** @endcond */

LIBSBML_CPP_NAMESPACE_END
