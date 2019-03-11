#include "models/cifar10_cnn.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "img_data.h"
#include <stdio.h>
#include <mbed.h>

static size_t argmax(S_TENSOR logits)
{
    float max_logit = logits->read<float>(0, 0);
    size_t max_label = 0;
    for (size_t i = 0; i < logits->getSize(); ++i)
    {
        float logit = logits->read<float>(0, i);
        if (logit > max_logit)
        {
            max_label = i;
            max_logit = logit;
        }
    }
    return max_label;
}

Serial pc(USBTX, USBRX, 115200);

int main(int argc, char *argv[])
{
    size_t num_imgs = sizeof(imgs_data) / sizeof(imgs_data[0]);
    printf("number of images: %lu\n", num_imgs);
    for (size_t label = 0; label < num_imgs; ++label)
    {
        Context ctx;
        float *data = images_data[label];
        Tensor *in_tensor = new WrappedRamTensor<float>({32, 32, 3}, data);
        get_cifar10_cnn_ctx(ctx, in_tensor);
        S_TENSOR logits = ctx.get("fully_connect_2/logits");
        ctx.eval();
        size_t pred_label = argmax(logits);
        printf("pred label: %lu, expecting %lu\n", pred_label, label);
    }
    return 0;
}
