#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

int main () {

  std::ifstream ifs ("test.txt", std::ifstream::in);

  char c = ifs.get();

  while (ifs.good()) {
    std::cout << c;
    c = ifs.get();
  }

  ifs.close();

  return 0;
}
