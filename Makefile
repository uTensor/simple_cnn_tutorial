compile: main.cpp
	mbed compile -m auto -t GCC_ARM --profile=uTensor/build_profile/release.json -f

compile-pc:
	g++ -D__ON_PC=1 -std=c++11 -I uTensor -I models main.cpp uTensor/uTensor/*/*.cpp models/*.cpp -o main