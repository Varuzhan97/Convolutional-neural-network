#ifndef __NORMAL_RANDOM_H__
#define __NORMAL_RANDOM_H__

#include <cmath>
#include <random>

namespace NormalRandom
{
// return a normally distributed random number
void normalRandom(double sigma, double mi, std::vector<std::vector<double>> & numbers);
}

#endif
