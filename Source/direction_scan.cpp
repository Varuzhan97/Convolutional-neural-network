#include "direction_scan.h"

#include <dirent.h>
#include <algorithm>
#include <memory>
#include <ctime>
//For back_inserter
#include <iterator>

std::vector<std::string> Scanner::scan_dir(const std::string& path, std::vector<int>& label) {

    DIR*    dir;
    dirent* pDir;
    std::vector<std::string> temp_files;
    dir = opendir(path.c_str());
    while (pDir = readdir(dir)) {
      std::string name = pDir->d_name;
      int32_t pos = name.find(".");
      std::string extension = name.substr(pos + 1);
      if (extension == "jpg") {
        std::string a = path+"/"+name;
        temp_files.push_back(a);
        //Get label
        char c = name[0];
        int ic = c-'0';
        label.push_back(ic);
      }
    }
    return temp_files;
}
