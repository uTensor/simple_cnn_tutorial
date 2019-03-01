#include "models/cifar10_cnn.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "img_data.h"
#include <stdio.h>
#include <mbed.h>
#include <string>

using namespace std;

int main(int argc, char *argv[])
{
    Context ctx;
    Tensor *in_tensor = new WrappedRamTensor<float>({32, 32, 3}, (float *)img_data);
    get_cifar10_cnn_ctx(ctx, in_tensor);
    S_TENSOR pred = ctx.get("pred:0");
    ctx.eval();
    int pred_label = *(pred->read<int>(0, 0));
    printf("\n");
    printf("pred label: %i, expecting %i\n",
           pred_label,
           label_true);
    return 0;
}
