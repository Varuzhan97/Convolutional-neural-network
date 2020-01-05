#include "pooling.h"
#include <iostream>
#include <algorithm>


Pooling::Pooling()
{

}

void Pooling::makeCache(const std::vector<std::vector<double>>& input)
{
  //Clear last cached and resize
  m_caschedInput.clear();
  m_caschedInput.resize(input.size());

  //Copy last input
  m_caschedInput.assign(input.begin(), input.end());
  m_cachedHeight = (m_outputHeight*2);
  m_cachedWidth = (m_outputHeight*2);
}

void Pooling::startPooling(const std::vector<std::vector<double>>& convVector, int height, int width)
{
  m_outputHeight = height/2;
  m_outputWidth = width/2;

  //Clear last input
  m_pooled.clear();

  //Starting process with all 8 inputs with (height X width) dimension
  makePooling( convVector, 0);
  makePooling( convVector, 1);
  makePooling( convVector, 2);
  makePooling( convVector, 3);
  makePooling( convVector, 4);
  makePooling( convVector, 5);
  makePooling( convVector, 6);
  makePooling( convVector, 7);

  //Make the cache of last input
  makeCache(convVector);
}

void Pooling::makePooling(const std::vector<std::vector<double>>& input, int index)
{
  std::vector<double> v;
  for(int i = 0; i < POOL_FILTER_STRIDE*m_outputHeight;i+=POOL_FILTER_STRIDE)
  {
    for (int j = 0; j < POOL_FILTER_STRIDE*m_outputWidth; j+=POOL_FILTER_STRIDE) {
      std::vector<double> temp;
      for (int k = 0; k < POOL_FILTER_HEIGHT; k++) {
        for (int n = 0; n < POOL_FILTER_WIDTH; n++) {
          temp.push_back(input[index][((i+k)*(m_outputWidth*2)+(j+n))]);
        }
      }
      double max = *max_element(temp.begin(), temp.end());
      v.push_back(max);
    }
  }
  m_pooled.push_back(v);

}

std::vector<std::vector<double>> Pooling::backProp(std::vector<std::vector<double>> d_L_d_out)
{
  //d_L_d_out is the loss gradient for this layer's outputs.
  std::vector<std::vector<double>> d_L_d_input;
  for (size_t i = 0; i < POOL_FILTER_NUMBER; i++) {
    std::vector<double> v;
    for (int j = 0; j < ((m_outputHeight*2)*(m_outputWidth*2)); j++)
      v.push_back(0);
    d_L_d_input.push_back(v);
  }

  for (int index = 0; index < POOL_FILTER_NUMBER; index++) {
    std::vector<double> v;
    int counter = 0;

    for(int i = 0; i < POOL_FILTER_STRIDE*m_outputHeight;i+=POOL_FILTER_STRIDE)
    {
      for (int j = 0; j < POOL_FILTER_STRIDE*m_outputWidth; j+=POOL_FILTER_STRIDE) {
        std::vector<double> temp;
        for (int k = 0; k < POOL_FILTER_HEIGHT; k++) {
          for (int n = 0; n < POOL_FILTER_WIDTH; n++) {
            temp.push_back(m_caschedInput[index][((i+k)*(m_outputWidth*2)+(j+n))]);
          }
        }
        double max = *max_element(temp.begin(), temp.end());
        bool p = true;
        for (int k = 0; k < POOL_FILTER_HEIGHT; k++) {
          for (int n = 0; n < POOL_FILTER_WIDTH; n++) {
            //If this pixel was the max value, copy the gradient to it.
            if((d_L_d_out[index][counter]==m_caschedInput[index][((i+k)*(m_outputWidth*2)+(j+n))] && max==m_caschedInput[index][((i+k)*m_width+(j+n))]) && p)
            {
              d_L_d_input[index][((i+k)*(m_outputWidth*2)+(j+n))] = max;
              p = false;
            }
            else
              d_L_d_input[index][((i+k)*(m_outputWidth*2)+(j+n))] = 0;
          }
        }
        counter++;
      }
    }
  }
  return d_L_d_input;
}

Pooling::~Pooling()
{

}
