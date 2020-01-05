#include <iostream>
#include "convolution.h"
#include "normal_random.h"

Convolution::Convolution()
{

}

void Convolution::makeCache(const std::vector<double>& input)
{
  //Clear last cached and resize
  m_cachedInput.clear();
  m_cachedInput.resize(input.size());

  //Copy last input
  m_cachedInput.assign(input.begin(), input.end());
}

void Convolution::startConvolution(const std::vector<double>& pixelVector, int height, int width)
{
  m_outputHeight = (height - 2);
  m_outputWidth = (width - 2);

  //For initialization of filters in first run
  if(p)
  {
    //Random number distribution that produces floating-point values according to a normal distribution
    NormalRandom::normalRandom(CONV_FILTER_NUMBER, CONV_FILTER_HEIGHT*CONV_FILTER_WIDTH, m_filters);

    //Diving by 9.0 during the initialization is important.
    //If the initial values are too large or too small, training the network will be ineffective
    for (size_t k = 0; k < CONV_FILTER_NUMBER; k++) {
      for (size_t i = 0; i < (CONV_FILTER_HEIGHT*CONV_FILTER_WIDTH); i++) {
          m_filters[k][i] = (double)m_filters[k][i]/(double)9.0;
      }
    }
    p = false;
  }

  //Clear last input
  m_conved.clear();

  //Starting process with all 8 filters
  makeConvolution( pixelVector, 0);
  makeConvolution( pixelVector, 1);
  makeConvolution( pixelVector, 2);
  makeConvolution( pixelVector, 3);
  makeConvolution( pixelVector, 4);
  makeConvolution( pixelVector, 5);
  makeConvolution( pixelVector, 6);
  makeConvolution( pixelVector, 7);

  //Make the cache of last input
  makeCache(pixelVector);
}

void Convolution::makeConvolution(const std::vector<double>& channel, int filterIndex)
{
  std::vector<double> v;
  for(int i = 0; i < m_outputHeight;i++)
  {
    for (int j = 0; j < m_outputWidth; j++) {
      double sum = 0;
      for (int k = 0; k < CONV_FILTER_HEIGHT; k++) {
        for (int n = 0; n < CONV_FILTER_WIDTH; n++) {
          //Storing pixel value into temp
          double temp = (channel[((i+k)*(m_outputWidth+2)+(j+n))]);
          //Storing ((pixel value)*(filter value)) into sum
          sum += (temp * m_filters[filterIndex][k*CONV_FILTER_WIDTH+n]);
        }
      }
      v.push_back(sum);
    }
  }
  m_conved.push_back(v);
}

void Convolution::backProp(std::vector<std::vector<double>> d_L_d_out, double learn_rate)
{
  //d_L_d_out is the loss gradient for this layer's outputs
  //filters with same shape as m_filters
  std::vector<std::vector<double>> filters;
  for (size_t i = 0; i < CONV_FILTER_NUMBER; i++) {
    std::vector<double> v;
    for (int j = 0; j < (CONV_FILTER_HEIGHT*CONV_FILTER_WIDTH); j++)
      v.push_back(0);
    filters.push_back(v);
  }

  //For keeping 3x3 reegions of last input
  std::vector<std::vector<double>> regions;

  //Loop for storing 3x3 regions into "regions"
  for(int i = 0; i < m_outputHeight;i++)
  {
    for (int j = 0; j < m_outputWidth; j++) {
      double sum = 0;
      std::vector<double> v;
      for (int k = 0; k < CONV_FILTER_HEIGHT; k++) {
        for (int n = 0; n < CONV_FILTER_WIDTH; n++) {
          v.push_back(m_cachedInput[((i+k)*(m_outputWidth+2)+(j+n))]);
        }
      }
      regions.push_back(v);
    }
  }

  //Loop for iterating d_L_d_out(last output of this layer)
  int counter = 0;
  for (int i = 0; i < m_outputHeight; i++) {
    for (int j = 0; j < m_outputWidth; j++) {
      //Loop for filters number
      for (size_t k = 0; k < 2; k++) {
        for (size_t m = 0; m < 3; m++) {
          for (size_t n = 0; n < 3; n++) {
            filters[k][m*3+n] += ((d_L_d_out[k][i*m_outputWidth+j]*regions[counter][m*3+n]));
          }
        }
      }
      counter++;
    }
  }

  for (size_t i = 0; i < CONV_FILTER_NUMBER; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 3; k++) {
        m_filters[i][j*3+k] -= (learn_rate * filters[i][j*3+k]);
      }
    }
  }
}


Convolution::~Convolution()
{

}
