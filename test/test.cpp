/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <cstring>
#include <gtest/gtest.h>

#include "gradient_test.hpp"
#include "training_test.hpp"

bool cattle::test::verbose;

int main(int argc, char** argv) {
	using cattle::test::verbose;
	static const char* verbose_flag = "-verbose";
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], verbose_flag)) {
			verbose = true;
			break;
		}
	}
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
