#include <string>
#include "cnn.h"

int main()
{
  std::string trainingPath = "PATH-TO-TRAINING-SET";
  std::string testingPath = "PATH-TO-TESTING-SET";
  CNN * cnn = new CNN(trainingPath, testingPath);
  //For high accuracy set more epochs instead of 3
  //Second parameter is learning rate
  cnn->runTrain(200, 0.003);
  cnn->runTest();
  delete cnn;
  return 0;
}
