/****************************************************************************
**
** Copyright (C) 2004-2005 Trolltech AS. All rights reserved.
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

#include <QApplication>
#include <QtGui>

#include "imagewindow.h"
#include "imagewindow_NOX.h"

int main(int argc, char *argv[])
{
//  std::ofstream logFile("cc3d-out.txt");
//  std::streambuf *outbuf = std::cerr.rdbuf(logFile.rdbuf());
//  std::streambuf *errbuf = std::cerr.rdbuf(logFile.rdbuf());
  
    Q_INIT_RESOURCE(player);
      
//     QCoreApplication *app;
//     QApplication app(argc, argv);
    bool silent=false;
    bool commandFlag=false;
    bool usePython=false;
    bool useXMLFileFlag=false;
    bool xServerFlag=true;
    

    QString xmlFileName;
    QString scrDesFileName;
    QString pythonScriptFileName;
    QString outputDirectory;
    bool noOutputFlag=false;
    

	 
    QRegExp regex_silent("^--silent");
    QRegExp regex_noXServer("^--noXServer");
    QRegExp regex_xml("^--xml=(.+)");
    QRegExp regex_scrDes("^--screenshotDescription=(.+)");
    QRegExp regex_pythonScript("^--pythonScript=(.+)");
    QRegExp regex_outputDirectory("^--outputDirectory=(.+)");

    QRegExp regex_noOutput("^--noOutput");

    int pos;
    for(int i =0 ; i< argc ;++i){
      QString argValue(argv[i]);
      pos=regex_silent.indexIn(argValue);
      if(pos>-1){
         silent=true;
         cerr<<"silent:"<<silent<<endl;
         
      }
      pos=regex_noXServer.indexIn(argValue);
      if(pos>-1){
         xServerFlag=false;
         cerr<<"noXServer: "<<!xServerFlag<<endl;
         
      }

      pos=regex_noOutput.indexIn(argValue);
      if(pos>-1){
         noOutputFlag=true;
         cerr<<"noOutputFlag: "<<noOutputFlag<<endl;
         
      }



      pos=regex_xml.indexIn(argValue);
      if(pos>-1){
         if(regex_xml.numCaptures()==1)
            xmlFileName=regex_xml.cap(1);
            cerr<<"xml:"<<xmlFileName.toStdString()<<endl;
            commandFlag=true;
            useXMLFileFlag=true;
            continue;
      }

      pos=regex_scrDes.indexIn(argValue);
      if(pos>-1){
         if(regex_scrDes.numCaptures()==1)
            scrDesFileName=regex_scrDes.cap(1);
            cerr<<"screenshotDescription:"<<scrDesFileName.toStdString()<<endl;
            continue;
      }

      pos=regex_outputDirectory.indexIn(argValue);
      if(pos>-1){
         if(regex_outputDirectory.numCaptures()==1)
            outputDirectory=regex_outputDirectory.cap(1);
            cerr<<"outputDirectory:"<<outputDirectory.toStdString()<<endl;
            continue;
      }

      
      pos=regex_pythonScript.indexIn(argValue);
      if(pos>-1){
         if(regex_pythonScript.numCaptures()==1)
            pythonScriptFileName=regex_pythonScript.cap(1);
            cerr<<"pythonScriptFileName:"<<pythonScriptFileName.toStdString()<<endl;
            usePython=true;
            commandFlag=true;
            continue;
      }
            
    }
     
    //silent mode of operation
    if(silent && !xServerFlag){
      QCoreApplication app(argc, argv);
//       app=new QCoreApplication(argc, argv);
//       cerr<<"GOT HERE before imageWin construction"<<endl;
      ImageWindow_NOX imageWin(true);
      imageWin.setXServerFlag(xServerFlag); //here we will set xServer flag it wil lbe used to decide how to output screenshots
//       cerr<<"GOT HERE after imageWin construction"<<endl;

		imageWin.setXMLFile("");
		imageWin.setPythonScript("");
		imageWin.setRunPythonFlag(false);


      if(useXMLFileFlag)
         imageWin.setXMLFile(xmlFileName);

      imageWin.setOutputDirectory(outputDirectory);

      if(noOutputFlag)
         imageWin.setNoOutputFlag(noOutputFlag);

      imageWin.setRunPythonFlag(false);
      if(usePython)
         imageWin.setPythonScript(pythonScriptFileName);

//       app.setMainWidget(&imageWin);

      if(!scrDesFileName.isEmpty())
         imageWin.setScreenshotDescriptionFileName(scrDesFileName);

      imageWin.updateSimulationFileNames();         

      imageWin.startSimulation();
      
      return app.exec();
    }

    if(silent && xServerFlag){
      QApplication app(argc, argv);
      ImageWindow imageWin(true);
      imageWin.resize(900, 500);
		
		imageWin.setXMLFile("");
		imageWin.setPythonScript("");
		imageWin.setRunPythonFlag(false);

      if(useXMLFileFlag)
         imageWin.setXMLFile(xmlFileName);
      imageWin.setOutputDirectory(outputDirectory);
      if(noOutputFlag)
         imageWin.setNoOutputFlag(noOutputFlag);
      imageWin.setRunPythonFlag(false);
      if(usePython)
         imageWin.setPythonScript(pythonScriptFileName);

      if(!scrDesFileName.isEmpty())
         imageWin.setScreenshotDescriptionFileName(scrDesFileName);

//       imageWin.show();
		imageWin.updateSimulationFileNames();
      imageWin.startSimulation();
      
      return app.exec();      

    }


    //command line operation with gui
    if(!silent && commandFlag){      
      QApplication app(argc, argv);
      ImageWindow imageWin;
      imageWin.resize(900, 500);
      cerr<<"      if(useXMLFileFlag)"<<useXMLFileFlag<<endl;
		//reset anythin that could be potentially read from file open dialog (xml script name etc...)
		imageWin.setXMLFile("");
		imageWin.setPythonScript("");
		imageWin.setRunPythonFlag(false);


      if(useXMLFileFlag){
         imageWin.setXMLFile(xmlFileName);
      }
      imageWin.setOutputDirectory(outputDirectory);
      if(noOutputFlag)
         imageWin.setNoOutputFlag(noOutputFlag);
      imageWin.setRunPythonFlag(false);

      if(usePython)
         imageWin.setPythonScript(pythonScriptFileName);

      if(!scrDesFileName.isEmpty())
         imageWin.setScreenshotDescriptionFileName(scrDesFileName);

      imageWin.updateSimulationFileNames();

      imageWin.show();
      app.setWindowIcon(QIcon(":/images/cc3d_64x64_logo.png"));
      imageWin.startSimulation();
      
      return app.exec();
    }
    QApplication app(argc, argv);
    ImageWindow imageWin;
//     app.setMainWidget(&imageWin);
    imageWin.resize(900, 500);
    imageWin.show();
    app.setWindowIcon(QIcon(":/images/cc3d_64x64_logo.png"));
    
    // restore the buffers
//    std::cerr.rdbuf(outbuf);
//    std::cerr.rdbuf(errbuf);
    
    return app.exec();

//     QApplication app(argc, argv);
//     ImageWindow imageWin;
//     imageWin.resize(800, 500);
//     imageWin.show();
//     return app.exec();
}
