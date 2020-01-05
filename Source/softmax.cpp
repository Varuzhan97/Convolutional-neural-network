#include "softmax.h"
#include <algorithm>
#include "normal_random.h"

void Softmax::makeFlatten(const std::vector<std::vector<double>>& input, int d)
{

  m_flatten.insert(m_flatten.end(), input[0].begin(), input[0].end());
  m_flatten.insert(m_flatten.end(), input[1].begin(), input[1].end());
  m_flatten.insert(m_flatten.end(), input[2].begin(), input[2].end());
  m_flatten.insert(m_flatten.end(), input[3].begin(), input[3].end());
  m_flatten.insert(m_flatten.end(), input[4].begin(), input[4].end());
  m_flatten.insert(m_flatten.end(), input[5].begin(), input[5].end());
  m_flatten.insert(m_flatten.end(), input[6].begin(), input[6].end());
  m_flatten.insert(m_flatten.end(), input[7].begin(), input[7].end());

}

Softmax::Softmax()
{

}

std::vector<double> Softmax::startSoftmax(const std::vector<std::vector<double>>& input, int h, int w)
{
  m_length = h*w*DEPTH;

  //For initialization of weights and biases in first run
  if(p)
  {
    //Assign 0 to m_biases(m_biase length is 10)
    m_biases.assign(NODES, 0.0);

    //Random number distribution that produces floating-point values according to a normal distribution
    NormalRandom::normalRandom(m_length, NODES, m_weights);

    for (size_t k = 0; k < m_length; k++) {
      for (size_t i = 0; i < NODES; i++) {
          m_weights[k][i] = (double)m_weights[k][i]/(double)m_length;
      }
    }
    p = false;
  }

  //Clear last flatten of input and final prediction of last time
  m_flatten.clear();
  m_total.clear();

  //Make input flatten
  makeFlatten(input, DEPTH);

  //Multiply flatten input and m_weights
  for (int i = 0; i < NODES; i++) {
    double sum = 0;
    //Loop for multiplying [j] weight for each digit [i] with each flatten [j]
    for (int j = 0; j < m_length; j++) {
      sum += (m_flatten[j]*m_weights[j][i]);
    }
    //Bias sum
    sum+=m_biases[i];
    m_total.push_back(sum);
  }

  std::vector<double> exponential;
  std::vector<double> predictions;

  double exp_sum = 0.0;
  double temp = 0.0;

  for(int i = 0; i < NODES; i++)
  {
    temp = exp(m_total[i]);
    exponential.push_back(temp);
    exp_sum+=(temp);
  }

  for(int i = 0; i < NODES; i++)
  {
    temp = ((double)exponential[i]/(double)exp_sum);
    predictions.push_back(temp);
  }

  //Make the cache of last input
  makeCache();

  return predictions;
}

Softmax::~Softmax()
{

}

void Softmax::makeCache()
{
  //Clear last cached and resize
  m_cachedFlatten.clear();
  m_cachedTotal.clear();
  m_cachedFlatten.resize(m_flatten.size());
  m_cachedTotal.resize(m_total.size());

  //Copy last input and parameters
  m_cachedLength = m_length;
  m_cachedFlatten.assign(m_flatten.begin(), m_flatten.end());
  m_cachedTotal.assign(m_total.begin(), m_total.end());

}

std::vector<std::vector<double>> Softmax::backProp(const std::vector<double>& d_L_d_out, const double learn_rate)
{
  //d_L_d_out is the loss gradient for this layer's outputs
  std::vector<std::vector<double>> d_L_d_inputs_shaped;

  for(int i = 0; i < NODES; i++)
  {
    //Only 1 element of d_L_d_out will be nonzero
    if(d_L_d_out[i]==0)
      continue;

    //Count e^totals & Sum of all e^totals
    std::vector<double> t_exp;
    double sum = 0.0;
    double temp = 0.0;
    for (size_t i = 0; i < NODES; i++) {
      temp = exp(m_cachedTotal[i]);
      t_exp.push_back(temp);
      sum+=(temp);
    }

    //Gradients of out[i] against totals
    std::vector<double> d_out_d_t;
    for (size_t j = 0; j < NODES; j++) {
      d_out_d_t.push_back((-t_exp[i]) * t_exp[j] / (double)(pow(sum,2)));
    }

    d_out_d_t[i] = t_exp[i] * (sum - t_exp[i]) / (double)(pow(sum,2));

    //Gradients of totals against weights/biases/input
    std::vector<double> d_t_d_w = m_cachedFlatten;

    double d_t_d_b = 1.0;

    std::vector<std::vector<double>> d_t_d_inputs = m_weights;

    //Gradients of loss against totals
    std::vector<double> d_L_d_t;
    for (size_t j = 0; j < NODES; j++) {
      d_L_d_t.push_back(d_L_d_out[i] * d_out_d_t[j]);
    }

    //Gradients of loss against weights/biases/input
    std::vector<std::vector<double>> d_L_d_w;
    for (int k = 0; k < m_cachedLength; k++) {
      std::vector<double> summator;
      for (int j = 0; j < NODES; j++) {
        double sum = 0;
        sum += (d_t_d_w[k]*d_L_d_t[j]);
        summator.push_back(sum);
      }
      d_L_d_w.push_back(summator);
    }

    std::vector<double> d_L_d_b;
    for (size_t j = 0; j < NODES; j++) {
      d_L_d_b.push_back(d_L_d_t[j] * d_t_d_b);
    }

    std::vector<double> d_L_d_inputs;
    for (int i = 0; i < m_cachedLength; i++) {
      double sum = 0.0;
      for (int j = 0; j < NODES; j++) {
        sum += (d_t_d_inputs[i][j]*d_L_d_t[j]);
      }
      d_L_d_inputs.push_back(sum);
    }

    //Update weights / biases
    for (int k = 0; k < m_weights.size(); k++)
    {
        for (int j = 0; j < m_weights[k].size(); j++)
        {
            m_weights[k][j] -= (learn_rate * d_L_d_w[k][j]);
        }
    }
    for (int k = 0; k < m_biases.size(); k++)
    {
      m_biases[k] -= (learn_rate * d_L_d_b[k]);
    }

    //We have to reshape() before returning d_L_d_inputs because we flattened the input during forward pass
    for (int k = 0; k < DEPTH; k++)
    {
      std::vector<double> input;
      for (size_t m = 0; m < (m_cachedLength/DEPTH); m++) {
        input.push_back(d_L_d_inputs[k*(m_cachedLength/DEPTH)+m]);
      }
      d_L_d_inputs_shaped.push_back(input);
    }

    return d_L_d_inputs_shaped;

  }
  return d_L_d_inputs_shaped;
}
