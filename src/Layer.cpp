/*
 * Layer.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include "Layer.h"
#include "Neuron.h"

Layer::Layer(int nodeCnt, Activation* act) : nodeCnt(nodeCnt), act(act), bias(0) {
	nodes = new Neuron[nodeCnt];
};

Layer::~Layer() {
	delete[] nodes;
	delete act;
}

