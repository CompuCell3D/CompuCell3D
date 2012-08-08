#pragma once
#include <fstream>
#include <streambuf>
#include <locale>

#include <QtGui>
#include <QtCore>

class QSender : public QObject
 {
     Q_OBJECT

 public:
     QSender() { m_value = 0; }

     int value() const { return m_value; }
	 void outputString(const QString& _str){
		emit(sendString(_str));
	 }

	 void emitZoom(int _factor){
		emit(valueChanged1(_factor));
	 }
	 QTextEdit * qePtr;

 public slots:

	 void setValue(int value){m_value=value;};
	 void ensureCursorVisible(const QString & _str){
		qePtr->ensureCursorVisible();
	 }

	 void colorText(const QString & _str){
		colorOrig=qePtr->textColor();
		qePtr->setTextColor(QColor("blue"));
	 }
	 void restoreOrigTextColor(const QString & _str){
		qePtr->setTextColor(colorOrig);
	 }
	
	 void redirectText(const QString & _str){
		qePtr->setTextColor(QColor("blue"));
		qePtr->insertPlainText(_str);
		qePtr->setTextColor(QColor("black"));
		qePtr->ensureCursorVisible();
	 }

 signals:
     void valueChanged(int newValue);
	 void valueChanged1(int newValue);
	 
	 void sendString(const QString & _str);


 private:
     int m_value;
	 QColor colorOrig;

 };