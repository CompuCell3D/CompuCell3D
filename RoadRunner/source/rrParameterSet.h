#ifndef rrParameterSetH
#define rrParameterSetH
#include "rrObject.h"
//---------------------------------------------------------------------------
namespace rr
{

class ParameterSets : public rrObject
{
    protected:

    public:


};


}

#endif

////using System;
////using System.Collections.Generic;
////using System.Xml;
////using libsbmlcs;
////
////namespace SBMLSupport
////{
////    public class ParameterSet
////    {
////        private string _Annotation;
////        private string _Id;
////        private string _MetaId;
////        private string _Name;
////        private string _Notes;
////        private Dictionary<string, double> _Values;
////
////        public void ApplyTo(Model model)
////        {
////            var keys = new string[_Values.Keys.Count];
////            _Values.Keys.CopyTo(keys, 0);
////            for (int i = 0; i < keys.Length; i++)
////            {
////                NOM.setValue(model, keys[i], _Values[keys[i]], false);
////            }
////        }
////        /// <summary>
////        /// Initializes a new instance of the ParameterSet class.
////        /// </summary>
////        public ParameterSet()
////        {
////            _Values = new Dictionary<string, double>();
////        }
////
////        public double this[int index]
////        {
////            get
////            {
////                var keys = new string[_Values.Keys.Count];
////                _Values.Keys.CopyTo(keys, 0);
////                return this[keys[index]];
////            }
////            set
////            {
////                var keys = new string[_Values.Keys.Count];
////                _Values.Keys.CopyTo(keys, 0);
////                this[keys[index]] = value;
////
////            }
////        }
////
////        public double this[string id]
////        {
////            get
////            {
////                return _Values[id];
////            }
////            set
////            {
////                _Values[id] = value;
////            }
////        }
////
////
////        public ParameterSet(Model model)
////            : this()
////        {
////            for (int i = 0; i < model.getNumCompartments(); i++)
////            {
////                Compartment current = model.getCompartment(i);
////                _Values[current.getId()] = current.getSize();
////            }
////
////            for (int i = 0; i < model.getNumSpecies(); i++)
////            {
////                Species current = model.getSpecies(i);
////                _Values[current.getId()] = current.getInitialConcentration();
////            }
////
////            for (int i = 0; i < model.getNumParameters(); i++)
////            {
////                Parameter current = model.getParameter(i);
////                _Values[current.getId()] = current.getValue();
////            }
////        }
////
////        /// <summary>
////        /// Initializes a new instance of the ParameterSet class.
////        /// </summary>
////        public ParameterSet(XmlElement element)
////            : this()
////        {
////            ReadFrom(element);
////        }
////
////        public string Notes
////        {
////            get { return _Notes; }
////            set { _Notes = value; }
////        }
////
////        public string Annotation
////        {
////            get { return _Annotation; }
////            set { _Annotation = value; }
////        }
////
////        public string Id
////        {
////            get { return _Id; }
////            set { _Id = value; }
////        }
////
////        public string Name
////        {
////            get { return _Name; }
////            set { _Name = value; }
////        }
////
////        public string MetaId
////        {
////            get { return _MetaId; }
////            set { _MetaId = value; }
////        }
////
////
////        public Dictionary<string, double> Values
////        {
////            get { return _Values; }
////            set { _Values = value; }
////        }
////
////        /// <summary>
////        /// Creates a new ParameterSet from a JDesigner jd2:parameterSet element
////        /// </summary>
////        /// <param name="element"></param>
////        /// <returns>null, in case an error occured, the parameter set otherwise</returns>
////        public static ParameterSet FromJD2Annotation(XmlElement element)
////        {
////            try
////            {
////                var set = new ParameterSet();
////                set.Id = element.GetAttribute("Id");
////
////                XmlNodeList notes = element.GetElementsByTagName("notes");
////                if (notes != null && notes.Count > 0)
////                    set.Notes = string.Format("<notes><body xmlns=\"http://www.w3.org/1999/xhtml\">{0}</body></notes>",
////                                              ((XmlElement) notes[0]).GetAttribute("note"));
////
////                XmlNodeList values = element.GetElementsByTagName("NameValue", "http://www.sys-bio.org/sbml");
////                if (values != null && values.Count > 0)
////                {
////                    foreach (object item in values)
////                    {
////                        var pair = (XmlElement) item;
////                        set.Values[pair.GetAttribute("name")] = Convert.ToDouble(pair.GetAttribute("value"));
////                    }
////                }
////
////                return set;
////            }
////            catch (Exception)
////            {
////                return null;
////            }
////        }
////
////
////        public void ReadFrom(XmlElement element)
////        {
////            _Id = element.GetAttribute("id");
////            _Name = element.GetAttribute("name");
////            _MetaId = element.GetAttribute("metaid");
////
////            XmlNodeList annotation = element.GetElementsByTagName("annotation");
////            if (annotation != null && annotation.Count > 0)
////            {
////                _Annotation = annotation[0].OuterXml;
////            }
////
////            XmlNodeList notes = element.GetElementsByTagName("notes");
////            if (notes != null && notes.Count > 0)
////            {
////                _Notes = notes[0].OuterXml;
////
////            }
////
////            XmlNodeList values = element.GetElementsByTagName("parameter");
////            foreach (object value in values)
////            {
////                var valueElement = (XmlElement) value;
////                _Values[valueElement.GetAttribute("sbmlId")] = Convert.ToDouble(valueElement.GetAttribute("value"));
////            }
////        }
////
////        public void WriteTo(XmlWriter writer)
////        {
////            writer.WriteStartElement("parameterSet", "http://sys-bio.org/ParameterSets");
////            if (!string.IsNullOrEmpty(_Id)) writer.WriteAttributeString("id", _Id);
////            if (!string.IsNullOrEmpty(_Name)) writer.WriteAttributeString("name", _Name);
////            if (!string.IsNullOrEmpty(_MetaId)) writer.WriteAttributeString("metaid", _MetaId);
////
////            if (!string.IsNullOrEmpty(_Annotation)) writer.WriteRaw(_Annotation);
////            if (!string.IsNullOrEmpty(_Notes))
////            {
////                if (!_Notes.Contains("<"))
////                    _Notes = string.Format("<notes><body xmlns=\"http://www.w3.org/1999/xhtml\">{0}</body></notes>",
////                                           _Notes);
////                writer.WriteRaw(_Notes);
////            }
////
////            foreach (string item in _Values.Keys)
////            {
////                string id = item;
////                double value = _Values[id];
////                if (string.IsNullOrEmpty(id)) continue;
////
////                writer.WriteStartElement("parameter", "http://sys-bio.org/ParameterSets");
////                writer.WriteAttributeString("sbmlId", id);
////                writer.WriteAttributeString("value", _Values[item].ToString());
////                writer.WriteEndElement();
////            }
////
////            writer.WriteEndElement();
////        }
////    }
////}

