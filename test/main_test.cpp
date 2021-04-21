/*
 * main_test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <gtest/gtest.h>

#include <cstring>

namespace cattle {
namespace test {

bool verbose;

}
}  // namespace cattle

int main(int argc, char** argv) {
  static const char* verbose_flag = "-verbose";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], verbose_flag)) {
      cattle::test::verbose = true;
      break;
    }
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
