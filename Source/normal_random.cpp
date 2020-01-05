#include "normal_random.h"
#include <iostream>
#include <chrono>

void NormalRandom::normalRandom(double sigma, double mi, std::vector<std::vector<double>>& numbers)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0.0,1.0);
  for(int i = 0; i < sigma; i++)
  {
    std::vector<double> temp;
    for (int j = 0; j < mi; j++) {
      double number = (distribution(generator));
      temp.push_back(number);
    }
    numbers.push_back(temp);
  }
}
