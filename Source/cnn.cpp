#include <iostream>
#include <string>

#include "cnn.h"
#include "direction_scan.h"


CNN::CNN(std::string trainPath, std::string testPath) : m_trainPath(trainPath), m_testPath(testPath)
{
  m_image = new Image();
  m_conv = new Convolution();
  m_pool = new Pooling();
  m_softmax = new Softmax();
}

CNN::~CNN()
{
  delete m_image;
  delete m_conv;
  delete m_pool;
  delete m_softmax;
}

std::vector<double> CNN::forward(int label, int& height, int& width)
{
  m_conv->startConvolution(m_image->getMergedChannel(), height, width);
  m_pool->startPooling(m_conv->getConvedMatrix(), m_conv->getHeight(), m_conv->getWidth());
  std::vector<double> pred = m_softmax->startSoftmax(m_pool->getPooled(), m_pool->getHeight(), m_pool->getWidth());

  m_loss = -log(pred[label]);

  int predIndex = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));

  if(predIndex == label)
    m_acc = 1;
  else
    m_acc = 0;

  return pred;
}

void CNN::train(int label, int& height, int& width, double& lr)
{
  std::vector<double> pred = forward(label, height, width);

  //Calculate initial gradient
  std::vector<double> gradient = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  gradient[label] = ((double)-1.0 / (double)pred[label]);

  std::vector<std::vector<double>> gradient_out = m_softmax->backProp(gradient, lr);
  std::vector<std::vector<double>> pool_out = m_pool->backProp(gradient_out);
  m_conv->backProp(pool_out, lr);
}

void CNN::runTrain(int trainEpoch, double learningRate)
{
  std::cout << "----------Training Netork----------" << '\n';
  std::cout << "Path----------" << m_trainPath << std::endl;
  std::vector<int> trainingLabels;
  std::vector<std::string> trainingFiles = Scanner::scan_dir(m_trainPath, trainingLabels);

  for (size_t epoch = 0; epoch < trainEpoch; epoch++) {

    int labelIndex = 0;
    double sumLoss = 0.0;
    double sumAcc = 0.0;

    for (std::vector<std::string>::iterator it = trainingFiles.begin() ; it != trainingFiles.end(); ++it)
    {
      std::string name = *it;
      int height = 0;
      int width = 0;
      m_image->loadImage(name, height, width);
      labelIndex = std::distance(trainingFiles.begin(), it);
      train(trainingLabels[labelIndex], height, width, learningRate);

      sumLoss += m_loss;
      sumAcc += m_acc;

    }
    labelIndex++;
    std::cout << "Epoch " << epoch << " : Average Loss " << (double)sumLoss / (double)labelIndex << " , Accuracy " << ((double)sumAcc/(double)labelIndex)*100 << " %" << '\n';
    sumLoss = 0.0;
    sumAcc = 0.0;
  }
}

void CNN::runTest()
{
  int labelIndex = 0;
  double sumLoss = 0.0;
  double sumAcc = 0.0;

  std::cout << "----------Testing Netork----------" << '\n';
  std::cout << "Path----------" << m_testPath << '\n';

  std::vector<int> testingLabels;
  std::vector<std::string> testingFiles = Scanner::scan_dir(m_testPath, testingLabels);

  int correct = 0;

  for (std::vector<std::string>::iterator it = testingFiles.begin() ; it != testingFiles.end(); ++it)
  {
    std::string name = *it;
    int height = 0;
    int width = 0;
    m_image->loadImage(name, height, width);
    labelIndex = std::distance(testingFiles.begin(), it);
    std::vector<double> out = forward(testingLabels[labelIndex], height, width);

    sumLoss += m_loss;
    sumAcc += m_acc;

    int predIndex = std::distance(out.begin(), std::max_element(out.begin(), out.end()));
    if(testingLabels[labelIndex]==predIndex)
      correct++;
      //std::cout << "*****Correct*****" << " Label " << testingLabels[labelIndex] << " prediction " << predIndex << '\n';
    //else
      //std::cout << "***** Wrong *****" << " Label " << testingLabels[labelIndex] << " prediction " << predIndex << '\n';
  }
  labelIndex++;
  std::cout << "Test results : Average Loss " << (double)sumLoss / (double)labelIndex << " , Accuracy " << ((double)sumAcc/(double)labelIndex)*100 << " %." << '\n';
  std::cout << "Images count " << labelIndex << " , Correct predicted " << correct << " , Wrong predicted " << labelIndex-correct << '\n';
}
