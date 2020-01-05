#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class Image
{

public:
  Image();
  ~Image();

  void loadImage(const std::string& stringPath, int& height, int& width);

  const std::vector<double>& getMergedChannel() const;
  void printMergedChannel();


private:
  std::vector<double> m_mergedVector;

  void generateChannels(Mat * image);

};

#endif
