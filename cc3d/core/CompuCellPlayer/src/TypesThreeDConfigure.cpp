/****************************************************************************
**
** Copyright (C) 2005-2005 Trolltech AS. All rights reserved.
**
** This file is part of the example classes of the Qt Toolkit.
**
** This file may be used under the terms of the GNU General Public
** License version 2.0 as published by the Free Software Foundation
** and appearing in the file LICENSE.GPL included in the packaging of
** this file.  Please review the following information to ensure GNU
** General Public Licensing requirements will be met:
** http://www.trolltech.com/products/qt/opensource.html
**
** If you are unsure which license is appropriate for your use, please
** review the following information:
** http://www.trolltech.com/products/qt/licensing.html or contact the
** sales department at sales@trolltech.com.
**
** This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
** WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
**
****************************************************************************/

#include <QtGui>

#include "TypesThreeDConfigure.h"
#include <StringUtils.h>
#include <vector>

using namespace std;

TypesThreeDConfigureForm::TypesThreeDConfigureForm(QWidget *parent)
    : QDialog(parent)
{
    ui.setupUi(this);
    static QRegExp regExp("(([0-9])+[,])+");
    ui.typeListEdit->setValidator(new QRegExpValidator(regExp,this));
}


void TypesThreeDConfigureForm::loadCurrentValues(std::vector<ushort > & types3DInvisibleVec){
   QString typeList;
   QString emptyStr;
   
   for(unsigned int i = 0 ; i < types3DInvisibleVec.size() ; ++i ){
//   for(set<ushort>::iterator sitr =types3DInvisibleSet ; i < types3DInvisibleSet.end() ; ++sitr ){
      
      typeList+=emptyStr.setNum(types3DInvisibleVec[i]);

      if( i != types3DInvisibleVec.size()-1)      
         typeList+=",";
   }
   ui.typeListEdit->setText(typeList);
}

void TypesThreeDConfigureForm::fillTypes3DInvisibleVec(std::vector<ushort> & types3DInvisibleVec){
      using namespace std;
       std::vector<string> vecTypesString;
   //    std::vector<unsigned short> vecTypesUShort;
      string typeList;
      typeList=ui.typeListEdit->text().toStdString().c_str();

      //remove trailing ,
      if(typeList[typeList.size()-1]==',')
         typeList=string(typeList,0,typeList.size()-1);

      //cerr<<"typeList="<<typeList<<endl;
      parseStringIntoList(typeList, vecTypesString ,",");
      //cerr<<"after parseStringIntoList  vecTypesString.size()="<<vecTypesString.size()<<endl;

/*      for(int i = 0  ; i < vecTypesString.size() ; ++i){

         cerr<<"vec["<<i<<"]="<<vecTypesString[i]<<endl;
      }*/

      types3DInvisibleVec.clear();
      for(int i = vecTypesString.size()-1 ; i >=0 ; --i ){
     //for(int i = 0 ; i < vecTypesString.size() ; ++i ){
     //cerr<<"vec_short["<<i<<"]="<<QString(vecTypesString[i]).toUShort()<<endl;


      types3DInvisibleVec.push_back(QString(vecTypesString[i].c_str()).toUShort());
     //types3DInvisibleSet.insert(QString(vecTypesString[i]).toUShort());
     //vecTypesUShort.push_back(QString(vecTypesString[i]).toUShort());

      }

//      sort(vecTypesUShort.begin(),vecTypesUShort.end());
//      types3DInvisibleVec.assign()
//      for(int i = 0 ; i < vecTypesUShort.size() ; ++i ){
//
//      }
  }

