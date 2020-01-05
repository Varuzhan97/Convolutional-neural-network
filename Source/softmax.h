#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include <vector>
#include "pooling.h"

//Macros for softmax layer
#define NODES 10
#define DEPTH 8

class Softmax {
public:

  Softmax();
  ~Softmax();

  std::vector<double> startSoftmax(const std::vector<std::vector<double>>& input, int h, int w);

  //Backpropagation
  std::vector<std::vector<double>> backProp(const std::vector<double>& d_L_d_out, const double learn_rate);

private:
  //Make input(H x W x D) vector flatten
  void makeFlatten(const std::vector<std::vector<double>>& input, int d);

  int m_length;

  //Layer weights vector
  std::vector<std::vector<double>> m_weights;

  //Layer biases vector
  std::vector<double> m_biases;

  //Vector for storing input as flatten vector
  std::vector<double> m_flatten;

  //Vector for storing final predictions for each node(digit)
  std::vector<double> m_total;

  //Members for making caches(it's for backpropagation)
  int m_cachedLength;
  std::vector<double> m_cachedFlatten;
  std::vector<double> m_cachedTotal;

  //Function for making caches of last input
  void makeCache();

  bool p = true;

};

#endif
