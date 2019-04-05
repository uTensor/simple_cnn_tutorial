# An End-to-End Tutorial Running CNN on MCU with uTensor

## Setup

1. `python2.7 -m virtualenv .venv`
2. `source .venv/bin/activate`
3. `pip install mbed-cli && mbed deploy`
4. `pip install utensor_cgen==0.3.3.dev2`

## Compilation

1. attach your board to the computer
2. run `make compile`

For the detailed guide, please refer to this [post](https://medium.com/@dboyliao/simple-cnn-on-mcu-with-utensor-372265ecc5b4)