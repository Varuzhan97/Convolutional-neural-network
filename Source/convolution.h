#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include <vector>
#include "image.h"

//Macros for convolution layer filters
#define CONV_FILTER_STRIDE 1
#define CONV_FILTER_HEIGHT 3
#define CONV_FILTER_WIDTH 3
#define CONV_FILTER_NUMBER 8

class Convolution
{

public:
  Convolution();
  ~Convolution();

  void startConvolution(const std::vector<double>& pixelVector, int height, int width);

  //Returns height and width of output
  int getHeight() {return m_outputHeight;}
  int getWidth() {return m_outputWidth;}

  //Returns result of layer
  const std::vector<std::vector<double>>& getConvedMatrix() const {return m_conved;}

private:
  //Vector for storing output values after convolution layer
  std::vector<std::vector<double>> m_conved;

  //Vector for storing filters generated values
  std::vector<std::vector<double>> m_filters;

  int m_outputHeight, m_outputWidth;

  void makeConvolution(const std::vector<double>& channel, int filterIndex);


  //Members for making caches(it's for backpropagation)
  std::vector<double> m_cachedInput;
  void makeCache(const std::vector<double>& input);
  bool p = true;

public:
  //Backpropagation
  void backProp(std::vector<std::vector<double>> d_L_d_out, double learn_rate);
};

#endif
