#ifndef __CNN_H__
#define __CNN_H__

#include <vector>
#include "image.h"
#include "convolution.h"
#include "pooling.h"
#include "softmax.h"

class CNN
{

public:
  CNN(std::string trainPath, std::string testPath);
  ~CNN();

  void runTrain(int trainEpoch = 1, double learningRate = 0.005);
  void runTest();

private:
  //Paths to training and test datasets
  std::string m_trainPath;
  std::string m_testPath;

  //Loss and accuracy
  double m_loss = 0.0;
  double m_acc = 0.0;

  Image * m_image;
  Convolution * m_conv;
  Pooling * m_pool;
  Softmax * m_softmax;

  std::vector<double> forward(int label, int& height, int& width);
  void train(int label, int& height, int& width, double& lr);
};

#endif
