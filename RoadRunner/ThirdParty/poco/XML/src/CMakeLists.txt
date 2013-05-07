CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(poco_xml_src)
set(target poco_xml) 

# Option that allows users to build static and/or shared libraries
# ----------------------------------------------------------------
OPTION(POCO_BUILD_STATIC_LIBS "Build static libraries" ON)
OPTION(POCO_BUILD_SHARED_LIBS "Build shared libraries" OFF)

set(sources 
AbstractContainerNode
AbstractNode
Attr
AttrMap
Attributes
AttributesImpl
CDATASection
CharacterData
ChildNodesList
Comment
ContentHandler
DOMBuilder
DOMException
DOMImplementation
DOMObject
DOMParser
DOMSerializer
DOMWriter
DTDHandler
DTDMap
DeclHandler
DefaultHandler
Document
DocumentEvent
DocumentFragment
DocumentType
Element
ElementsByTagNameList
Entity
EntityReference
EntityResolver
EntityResolverImpl
ErrorHandler
Event
EventDispatcher
EventException
EventListener
EventTarget
InputSource
LexicalHandler
Locator
LocatorImpl
MutationEvent
Name
NamePool
NamedNodeMap
NamespaceStrategy
NamespaceSupport
Node
NodeFilter
NodeIterator
NodeList
Notation
ParserEngine
ProcessingInstruction
SAXException
SAXParser
Text
TreeWalker
WhitespaceFilter
XMLException
XMLFilter
XMLFilterImpl
XMLReader
XMLString
XMLWriter
NodeAppender
xmlparse
xmlrole
xmltok

)

if(${BORLAND})
add_definitions(
             -w-8012                     #Comparing signed /unsigned
             -w-8057                     #Parameter never used
             -w-8004                     #'var' is assigned a value that is never used
             -w-8123                     #Path not found
             -w-8066                     #Unreachable code
             -w-8104                     #Local static with constructor...
             -w-8008                     #Condition is always false..
        )
endif()

add_definitions( -D_WIN32 -DXML_NS -DXML_DTD -DHAVE_EXPAT_CONFIG_H -DXML_STATIC)

#if(CMAKE_SYSTEM MATCHES "Windows")
# 
#endif(CMAKE_SYSTEM MATCHES "Windows")
#
#if (CMAKE_SYSTEM MATCHES "Linux")
#  add_definitions( -DPOCO_OS_FAMILY_UNIX )
#  # Standard 'must be' defines
#  add_definitions( -D_XOPEN_SOURCE=500 -D_REENTRANT -D_THREAD_SAFE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64)
#  set(SYSLIBS  pthread dl rt)
#endif(CMAKE_SYSTEM MATCHES "Linux")

include_directories(
../../Foundation/include
../../XML/include
)

#Then the static version
add_library(${target}-static STATIC ${sources})

#set_property(TARGET ${target}-static  
#PROPERTY  COMPILE_DEFINITIONS 
#POCO_STATIC
#PCRE_STATIC
#) 
#
#if(WIN32)
#set_property(TARGET ${target}-static  
#PROPERTY COMPILE_DEFINITIONS 
#WIN32
#) 
#
#endif()
#
#INSTALLING ======================
set(poco_xml_include_folder ${CMAKE_CURRENT_SOURCE_FOLDER}../include/Poco/XML)
set(poco_dom_include_folder ${CMAKE_CURRENT_SOURCE_FOLDER}../include/Poco/DOM)
set(poco_sax_include_folder ${CMAKE_CURRENT_SOURCE_FOLDER}../include/Poco/SAX)

FILE (GLOB xml_headers ${poco_xml_include_folder}/*.h)
FILE (GLOB dom_headers ${poco_dom_include_folder}/*.h)
FILE (GLOB sax_headers ${poco_sax_include_folder}/*.h)

#Install required files
install(FILES ${xml_headers} DESTINATION include/Poco/XML)
install(FILES ${dom_headers} DESTINATION include/Poco/DOM)
install(FILES ${sax_headers} DESTINATION include/Poco/SAX)

install(TARGETS ${target}-static 
LIBRARY DESTINATION lib 
ARCHIVE DESTINATION lib
)

