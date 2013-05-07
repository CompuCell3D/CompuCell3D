#ifndef rrSymbolListH
#define rrSymbolList
#include <vector>
#include "rrObject.h"
#include "rrSymbol.h"

using std::vector;

namespace rr
{

class RR_DECLSPEC SymbolList : public rrObject, public vector<Symbol> //Using vector instead of list since accessing element by []
{
    public:
        void                     Clear();
        int                      Add(const Symbol& item);
        double                   getValue(const int& index);
        string                   getName(const int& index);
        string                   getKeyName(const int& index);
        bool                     find(const string& name, int& index);
        bool                     find(const string& keyName, const string& name, int& index);
        unsigned int             Count(){return size();}
}; //class

}//namespace rr

#endif

//c#
//using System.Collections;
//
///* Filename    : symbolList.cs
// * Description : .NET based simulator
// * Author(s)   : Herbert M Sauro
// * Organization: Keck Graduate Institute
// * Created     : 2005
// *
// * This software is licenced under the BSD Licence.
// *
// * Original author:
// *   Herbert Sauro (hsauro@kgi.edu)
// *
// * Copyright (c) 2005 <Herbert M Sauro>
// * All rights reserved
// *
// * Redistribution and use in source and binary forms, with or without
// *  modification, are permitted provided that the following conditions
// * are met:
// * 1. Redistributions of source code must retain the above copyright
// *    notice, this list of conditions and the following disclaimer.
// * 2. Redistributions in binary form must reproduce the above copyright
// *    notice, this list of conditions and the following disclaimer in the
// *    documentation and/or other materials provided with the distribution.
// * 3. The name of the author may not be used to endorse or promote products
// *    derived from this software without specific prior written permission.
//
// * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// *****************************************************************************/
//
//namespace LibRoadRunner.Util
//{
//    public class SymbolList : ArrayList
//    {
//        /// <summary>
//        /// Returns the Symbol by the given index
//        /// </summary>
//        /// <param name="index">the index of the element to return</param>
//        /// <returns>Returns the Symbol by the given index</returns>
//        public new Symbol this[int index]
//        {
//            get { return ((Symbol) base[index]); }
//        }
//
//        /// <summary>
//        /// Returns the Symbol with the given Name
//        /// </summary>
//        /// <param name="sName">the name of the symbol to return</param>
//        /// <returns>Returns the Symbol with the given Name or null if not found</returns>
//        public Symbol this[string sName]
//        {
//            get
//            {
//                int nIndex = -1;
//                if (find(sName, out nIndex))
//                {
//                    return this[nIndex];
//                }
//                return null;
//            }
//        }
//
//        public int Add(Symbol item)
//        {
//            return base.Add(item);
//        }
//
//        public double getValue(int index)
//        {
//            return ((Symbol) base[index]).value;
//        }
//
//        public string getName(int index)
//        {
//            return ((Symbol) base[index]).name;
//        }
//
//        public string getKeyName(int index)
//        {
//            return ((Symbol) base[index]).keyName;
//        }
//
//        public override void Clear()
//        {
//            Clear();
//        }
//
//        public bool find(string name, out int index)
//        {
//            index = -1;
//            Symbol sym;
//            for (int i = 0; i < Count; i++)
//            {
//                sym = (Symbol) base[i];
//                if (name == sym.name)
//                {
//                    index = i;
//                    return true;
//                }
//                if (sym.name == name)
//                {
//                    index = i;
//                    return true;
//                }
//            }
//            return false;
//        }
//
//        public bool find(string keyName, string name, out int index)
//        {
//            index = -1;
//            for (int i = 0; i < Count; i++)
//            {
//                var sym = (Symbol) base[i];
//                if ((sym.name == name) && (sym.keyName == keyName))
//                {
//                    index = i;
//                    return true;
//                }
//            }
//            return false;
//        }
//
////        public bool find (string name, out int index) {
////            index = -1;
////            TSymbol sym;
////            for (int i=0; i<this.Count; i++) {
////                sym = (TSymbol) base[i];
////                if (name.ToUpper() == sym.name.ToUpper()) {
////                   index = i;
////                   return true;
////                }
////                if (sym.name.ToUpper() == name.ToUpper()) {
////                    index = i;
////                    return true;
////                }
////            }
////            return false;
////        }
////
////        public bool find (string keyName, string name, out int index) {
////            index = -1;
////            for (int i=0; i<this.Count; i++) {
////                TSymbol sym = (TSymbol) base[i];
////                if ((sym.name.ToUpper() == name.ToUpper()) && (sym.keyName.ToUpper() == keyName.ToUpper())) {
////                    index = i;
////                    return true;
////                }
////            }
////            return false;
////        }
//    }
//}
