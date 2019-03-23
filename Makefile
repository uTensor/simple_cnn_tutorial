compile: img_data.h main.cpp model
	mbed compile -m auto -t GCC_ARM --profile=uTensor/build_profile/release.json -f

compile-pc: img_data.h main.cpp model
	g++ -g -D__ON_PC=1 -std=c++11 -I uTensor -I models main.cpp uTensor/uTensor/*/*.cpp models/*.cpp -o main

img_data.h:
	.venv/bin/python prepare_test_data.py 

model: models/cifar10_cnn.cpp

models/cifar10_cnn.cpp: cifar10_cnn.pb
	utensor-cli convert cifar10_cnn.pb --output-nodes=fully_connect_2/logits

.PHONY: clean

clean:
	rm -rf *.dSYM ./main
