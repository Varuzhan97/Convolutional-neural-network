#include <iostream>
#include <string>
#include "image.h"

Image::Image()
{

}

Image::~Image()
{

}

void Image::loadImage(const std::string& stringPath, int& height, int& width)
{
  //Create vector
  m_mergedVector.clear();

  //Creates new Mat for processing the image
  Mat * image = new Mat();

  //Read the file
  (*image) = imread( stringPath, CV_LOAD_IMAGE_COLOR);

  //For returning image height and width
  height = image->rows;
  width = image->cols;

  //Generate RGB mixed channel
  this->generateChannels(image);
  delete image;
}


//For storing RGB channel into vectors
//(pixel-value/765)-0.5 is for transforming the image from 3 x [0, 255] to [-0.5, 0.5] to make it easier to work with
void Image::generateChannels(Mat * image)
{
  for(int i=0;i<image->rows;i++)
  {
      for(int j=0;j<image->cols;j++)
      {
        double r = ((double)image->at<Vec3b>(i,j)[2]);
        double g = ((double)image->at<Vec3b>(i,j)[1]);
        double b = ((double)image->at<Vec3b>(i,j)[0]);
        double sum = (r+g+b);
        sum = (((double)sum/(double)765)-0.5);
        m_mergedVector.push_back(sum);
      }
  }
}

const std::vector<double>& Image::getMergedChannel() const
{
  return m_mergedVector;
}
