#include "models/cifar10_cnn.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "img_data.h"
#include <stdio.h>
#include <mbed.h>
#include <string>

using namespace std;

Serial pc(USBTX, USBRX, 115200);

int main(int argc, char *argv[])
{
    printf("Simple CNN with uTensor!\n");
    Context ctx;
    printf("creating input tensor\n");
    Tensor *in_tensor = new WrappedRamTensor<float>({1, 32, 32, 3}, (float *)img_data);
    get_cifar10_cnn_ctx(ctx, in_tensor);
    printf("successfully build graph\n");
    S_TENSOR logits = ctx.get("fully_connect_2/logits:0");
    printf("evaluate prediction\n");
    ctx.eval();
    float max_value = *(logits->read<float>(0, 0));
    uint32_t pred_label = 0;
    for (uint32_t i = 0; i < logits->getSize(); ++i)
    {
        float value = *(logits->read<float>(0, 0) + i);
        if (value > max_value)
        {
            pred_label = i;
        }
    }
    printf("pred label: %lu, expecting %i\n", pred_label, label_true);
    return 0;
}
