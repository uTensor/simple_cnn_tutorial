compile: main.cpp
	mbed compile -m auto -t GCC_ARM --profile=uTensor/build_profile/release.json -f

compile-pc: img_data.h
	g++ -g -D__ON_PC=1 -std=c++11 -I uTensor -I models main.cpp uTensor/uTensor/*/*.cpp models/*.cpp -o main

img_data.h:
	.venv/bin/python prepare_test_data.py 

.PHONY: clean

clean:
	rm -rf *.dSYM ./main
