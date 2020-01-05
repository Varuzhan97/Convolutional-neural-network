#ifndef __POOLING_H__
#define __POOLING_H__

#include <vector>
#include "convolution.h"

//Macros for pooling layer filter
#define POOL_FILTER_STRIDE 2
#define POOL_FILTER_HEIGHT 2
#define POOL_FILTER_WIDTH 2
#define POOL_FILTER_NUMBER 8

class Pooling {

public:

  Pooling();
  ~Pooling();

  void startPooling(const std::vector<std::vector<double>>& convVector, int height, int width);

  //Returns height and width of output
  int getHeight() {return m_outputHeight;}
  int getWidth() {return m_outputWidth;}

  //Returns result of layer
  const std::vector<std::vector<double>>& getPooled() const {return m_pooled;}

  //Backpropagation
  std::vector<std::vector<double>> backProp(std::vector<std::vector<double>> d_L_d_out);

private:
  //Vector for storing output values after pooling layer
  std::vector<std::vector<double>> m_pooled;

  int m_height, m_width;
  int m_outputHeight, m_outputWidth;

  //Members for making caches(it's for backpropagation)
  std::vector<std::vector<double>> m_caschedInput;
  int m_cachedHeight, m_cachedWidth;

  //Function for making caches of last input
  void makeCache(const std::vector<std::vector<double>>& input);

  void makePooling(const std::vector<std::vector<double>>& input, int index);
};

#endif
