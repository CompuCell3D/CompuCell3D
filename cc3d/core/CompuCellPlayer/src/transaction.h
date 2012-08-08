#ifndef TRANSACTION_H
#define TRANSACTION_H
#include <QtGui>
#include <QApplication>
#include <map>
#include <vector>
#include "GraphicsData.h"
#include <CompuCell3D/plugins/PlayerSettings/PlayerSettings.h>
#include <iostream>
#include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>

class QWidget;

class Transaction
{
public:

	virtual void applySimulation(bool *_stopSimulation){}//TEMP
	//	virtual void applySimulation(){}
	virtual ~Transaction(){}

	//     void setTargetWidget(QWidget *widget){targetWidget=widget;}
	//     QWidget * getTargetWidget(){return targetWidget;}

	void setTargetObject(QObject *_object){
		//       using namespace std;

		targetObject=_object;
		//       cerr<<"transaction.h this is a target object="<<targetObject<<endl;
	}
	QObject * getTargetObject(){return targetObject;}

private:
	//    QWidget *targetWidget;
	QObject *targetObject;

};


class Projection2DData;

class CustomEvent:public QEvent{
public:
	CustomEvent():QEvent(QEvent::User){}
	virtual ~CustomEvent(){}
	//CustomEvent(const QEvent::Type & _type):QEvent(_type){};
	virtual QEvent::Type type()=0;
};

class TransactionStartEvent : public CustomEvent
{
public:
	enum Type{TransactionStart = 1001};
	TransactionStartEvent(const QEvent::Type & _type=static_cast<QEvent::Type> (TransactionStart))
	{ 
		latticeType=CompuCell3D::SQUARE_LATTICE;
	}

	virtual QEvent::Type type(){return static_cast<QEvent::Type> (TransactionStart);}
	int xSize, ySize, zSize;
	CompuCell3D::PlayerSettings playerSettings;
	int numSteps;
	//std::map<unsigned short,QPen> *typePenMapPtr;
	//std::map<unsigned short,QBrush> *typeBrushMapPtr;
	//Projection2DData *projDataPtr;
	QString message;
	CompuCell3D::LatticeType latticeType;

};


class Graphics2D;//forward declaration

class TransactionRefreshEvent : public CustomEvent
{
public:
	enum Type{TransactionRefresh = 1003};
	TransactionRefreshEvent(const QEvent::Type & _type=static_cast<QEvent::Type> (TransactionRefresh)){}
	virtual QEvent::Type type(){return static_cast<QEvent::Type> (TransactionRefresh);}
	//Graphics2D *graphics2DPtr;
	unsigned int mcStep;
	QString message;
};

class TransactionFinishEvent : public CustomEvent
{
public:
	enum Type{TransactionFinish = 1004};
	TransactionFinishEvent(const QEvent::Type & _type=static_cast<QEvent::Type> (TransactionFinish)):exitFlag(false){}
	virtual QEvent::Type type(){return static_cast<QEvent::Type> (TransactionFinish);}
	QString message;
	bool exitFlag;
};

class TransactionErrorEvent : public CustomEvent
{
public:
	enum Type{TransactionError = 1005};
	TransactionErrorEvent(const QEvent::Type & _type=static_cast<QEvent::Type> (TransactionError)):
	message("General error has occured"),
		errorCategory("General Error")
	{}
	virtual QEvent::Type type(){return static_cast<QEvent::Type> (TransactionError);}
	QString message;
	QString errorCategory;

};

class TransactionStopSimulationEvent : public CustomEvent
{
public:
	enum Type{TransactionStopSimulation = 1006};
	TransactionStopSimulationEvent(const QEvent::Type & _type=static_cast<QEvent::Type> (TransactionStopSimulation)):exitFlag(false){}
	virtual QEvent::Type type(){return static_cast<QEvent::Type> (TransactionStopSimulation);}
	QString message;
	bool exitFlag;
};


#endif
