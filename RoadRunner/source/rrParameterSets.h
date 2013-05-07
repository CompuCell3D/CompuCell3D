#ifndef rrParameterSetsH
#define rrParameterSetsH
#include "rrObject.h"
//---------------------------------------------------------------------------
namespace rr
{
class ParameterSets : public rrObject
{


};


}
#endif

////using System;
////using System.Collections.Generic;
////using System.IO;
////using System.Text;
////using System.Xml;
////using libsbmlcs;
////
////namespace SBMLSupport
////{
////    public class ParameterSets
////    {
////        private string _ActiveId;
////
////        private List<ParameterSet> _ParameterSets;
////
////        public ParameterSets()
////        {
////            _ParameterSets = new List<ParameterSet>();
////        }
////
////        public ParameterSets(string sbmlContent)
////            : this()
////        {
////            ReadParameterSets(sbmlContent);
////        }
////
////        public string ActiveId
////        {
////            get { return _ActiveId; }
////            set { _ActiveId = value; }
////        }
////
////        public List<ParameterSet> Sets
////        {
////            get { return _ParameterSets; }
////            set { _ParameterSets = value; }
////        }
////
////        public ParameterSet this[int index]
////        {
////            get { return _ParameterSets[index]; }
////        }
////
////        public ParameterSet this[string id]
////        {
////            get
////            {
////                foreach (ParameterSet item in _ParameterSets)
////                {
////                    if (item.Id == id) return item;
////                }
////
////                throw new Exception(string.Format("No parameter set exists with the given id '{0}'.", id));
////            }
////        }
////
////        public ParameterSet ActiveSet
////        {
////            get { return this[_ActiveId]; }
////        }
////
////        public static ParameterSets FromSBML(string sbmlContent)
////        {
////            return new ParameterSets(sbmlContent);
////        }
////
////        public static ParameterSets FromFile(string fileName)
////        {
////            return new ParameterSets(File.ReadAllText(fileName));
////        }
////
////
////        private void ReadJD2Sets(XmlDocument doc)
////        {
////            XmlNodeList jd2Sets = doc.DocumentElement.GetElementsByTagName("listOfParameterSets",
////                                                                           "http://www.sys-bio.org/sbml");
////            if (jd2Sets.Count > 0)
////            {
////                var parameterSets = (XmlElement) jd2Sets[0];
////                XmlNodeList sets = parameterSets.GetElementsByTagName("parameterSet", "http://www.sys-bio.org/sbml");
////                _ActiveId = parameterSets.GetAttribute("currentParameterSet");
////                foreach (object item in sets)
////                {
////                    ParameterSet jd2Set = ParameterSet.FromJD2Annotation((XmlElement) item);
////                    if (jd2Set != null)
////                        _ParameterSets.Add(jd2Set);
////                }
////            }
////        }
////
////        public void ReadParameterSets(XmlElement parameterSets)
////        {
////            XmlNodeList sets = parameterSets.GetElementsByTagName("parameterSet", "http://sys-bio.org/ParameterSets");
////            _ActiveId = parameterSets.GetAttribute("active");
////            foreach (object item in sets)
////            {
////                var newSet = new ParameterSet((XmlElement)item);
////                _ParameterSets.Add(newSet);
////            }
////        }
////
////        private void ReadSets(XmlDocument doc)
////        {
////            XmlNodeList allSets = doc.DocumentElement.GetElementsByTagName("listOfParameterSets",
////                                                                           "http://sys-bio.org/ParameterSets");
////            if (allSets.Count > 0)
////            {
////                var parameterSets = (XmlElement) allSets[0];
////                ReadParameterSets(parameterSets);
////            }
////        }
////
////        private void ReadParameterSets(string sbmlContent)
////        {
////            _ParameterSets.Clear();
////
////            var doc = new XmlDocument();
////            doc.LoadXml(sbmlContent);
////
////            ReadJD2Sets(doc);
////            ReadSets(doc);
////        }
////
////
////        public void WriteTo(XmlWriter writer)
////        {
////            writer.WriteStartElement("listOfParameterSets", "http://sys-bio.org/ParameterSets");
////            writer.WriteAttributeString("active", _ActiveId);
////            foreach (var item in _ParameterSets)
////            {
////                item.WriteTo(writer);
////            }
////            writer.WriteEndElement();
////        }
////
////        public string ToXmlString()
////        {
////            var builder = new StringBuilder();
////            var writer = XmlWriter.Create(builder, new XmlWriterSettings { Indent = true, OmitXmlDeclaration = true, Encoding = Encoding.UTF8 });
////            writer.WriteStartDocument();
////            WriteTo(writer);
////            writer.WriteEndDocument();
////            writer.Flush();
////            writer.Close();
////            return builder.ToString();
////        }
////
////        public void AddToModel(Model model)
////        {
////            if (model == null) return;
////            if (_ParameterSets.Count == 0) return;
////
////            var annotations = model.getAnnotationString();
////
////            if (string.IsNullOrEmpty(annotations))
////            {
////                model.setAnnotation("<annotation>" + ToXmlString() + "</annotation>");
////            }
////            else
////            {
////                // parse annotation
////                var annotationDoc = new XmlDocument();
////                var stream = new MemoryStream(Encoding.UTF8.GetBytes(annotations));
////
////                var ns = new XmlNamespaceManager(annotationDoc.NameTable);
////                ns.AddNamespace("jd2", "http://www.sys-bio.org/sbml");
////
////                XmlParserContext pc = new XmlParserContext(null, ns, string.Empty, XmlSpace.Default);
////                XmlTextReader reader = new XmlTextReader(stream, XmlNodeType.Element, pc);
////
////                annotationDoc.Load(reader);
////                //annotationDoc.LoadXml(annotations);
////
////                var parameterDoc = new XmlDocument();
////                parameterDoc.LoadXml(ToXmlString());
////
////                // remove possibly existing parameter set
////                var existingAnnotation = annotationDoc.DocumentElement.GetElementsByTagName("listOfParameterSets", "http://sys-bio.org/ParameterSets");
////                if (existingAnnotation.Count > 0)
////                {
////                    var elementToDelete = (XmlElement)existingAnnotation[0];
////                    var parent = elementToDelete.ParentNode;
////                    parent.RemoveChild(elementToDelete);
////                }
////
////                // add new node
////
////                var root = annotationDoc.DocumentElement;
////                var import = annotationDoc.ImportNode(parameterDoc.DocumentElement, true);
////                root.AppendChild(import);
////
////                // set the new annotations
////
////                var builder = new StringBuilder();
////                var writer = XmlWriter.Create(builder, new XmlWriterSettings { Indent = true, OmitXmlDeclaration = true, Encoding = Encoding.UTF8 });
////                writer.WriteStartDocument();
////                annotationDoc.DocumentElement.WriteTo(writer);
////                writer.WriteEndDocument();
////                writer.Flush();
////                writer.Close();
////
////                model.setAnnotation(builder.ToString());
////
////
////            }
////
////
////        }
////
////        //public static string TestAppend()
////        //{
////        //    string fileName = @"C:\Users\fbergmann\Desktop\jd2parameterSet.xml";
////        //    var sets = SBMLSupport.ParameterSets.FromFile(fileName);
////        //    var doc = libsbml.libsbml.readSBMLFromString(File.ReadAllText(fileName));
////        //    if (doc.getModel() != null)
////        //        sets.AddToModel(doc.getModel());
////        //    var newSBML = libsbml.libsbml.writeSBMLToString(doc);
////        //    return newSBML;
////        //}
////
////
////    }
////}

