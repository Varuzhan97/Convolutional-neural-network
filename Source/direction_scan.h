#ifndef __DIRECTION_SCAN_H__
#define __DIRECTION_SCAN_H__

#include <vector>
#include <string>

namespace Scanner
{
  std::vector<std::string> scan_dir(const std::string& path,std::vector<int>& label);
}

#endif
