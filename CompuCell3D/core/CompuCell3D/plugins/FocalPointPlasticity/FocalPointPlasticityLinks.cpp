#include <CompuCell3D/CC3D.h>

#include "FocalPointPlasticityLinks.h"

using namespace CompuCell3D;

/**
Written by T.J. Sego, Ph.D.
*/

void FocalPointPlasticityLinkBase::initializeConstitutiveLaw(std::string _localLaw) {
	ev = ExpressionEvaluator();
	ev.addVariable("Lambda");
	ev.addVariable("Length");
	ev.addVariable("TargetLength");
	ev.setExpression(_localLaw);
}

void FocalPointPlasticityLinkBase::setConstitutiveLaw(std::string _lawString) {
	initializeConstitutiveLaw(_lawString);
	usingLocalLaw = true;
}

double FocalPointPlasticityLinkBase::constitutiveLaw(float _lambda, float _length, float _targetLength) {
	ev[0] = _lambda;
	ev[1] = _length;
	ev[2] = _targetLength;
	return ev.eval();
}

float FocalPointPlasticityLinkBase::getDistance() {
	float xCM1 = initiator->xCM / initiator->volume;
	float yCM1 = initiator->yCM / initiator->volume;
	float zCM1 = initiator->zCM / initiator->volume;
	float xCM2, yCM2, zCM2;
	if (fppltd.anchor) {
		xCM2 = fppltd.anchorPoint[0];
		yCM2 = fppltd.anchorPoint[1];
		zCM2 = fppltd.anchorPoint[2];
	}
	else {
		xCM2 = initiated->xCM / initiated->volume;
		yCM2 = initiated->yCM / initiated->volume;
		zCM2 = initiated->zCM / initiated->volume;
	}

	return (float)distInvariantCM(xCM1, yCM1, zCM1, xCM2, yCM2, zCM2, potts->getCellFieldG()->getDim(), BoundaryStrategy::getInstance());
}

float FocalPointPlasticityLinkBase::getTension() {
	return fppltd.lambdaDistance * (getDistance() - fppltd.targetDistance);
}

std::vector<CellG*> FocalPointPlasticityLink::getCellPair() {
	std::vector<CellG*> o = std::vector<CellG*>(2);
	o[0] = getObj0();
	o[1] = getObj1();
	return o;
}

std::vector<CellG*> FocalPointPlasticityInternalLink::getCellPair() {
	std::vector<CellG*> o = std::vector<CellG*>(2);
	o[0] = getObj0();
	o[1] = getObj1();
	return o;
}

