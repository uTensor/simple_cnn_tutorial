#include "models/cifar10_cnn.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "img_data.h"
#include <stdio.h>
#ifndef __ON_PC
#include <mbed.h>
#endif

static size_t argmax(S_TENSOR logits)
{
    float max_logit = *(logits->read<float>(0, 0));
    size_t max_label = 0;
    for (size_t i = 0; i < logits->getSize(); ++i)
    {
        float logit = *(logits->read<float>(i, 0));
        if (logit > max_logit)
        {
            max_label = i;
            max_logit = logit;
        }
    }
    return max_label;
}

#ifndef __ON_PC
Serial pc(USBTX, USBRX, 115200);
#endif

int main(int argc, char *argv[])
{
    size_t num_imgs = sizeof(imgs_data) / sizeof(imgs_data[0]);
    printf("number of images: %lu\n", num_imgs);
    float acc = 0;
    for (size_t label = 0; label < num_imgs; ++label)
    {
        Context ctx;
        float *data = &(imgs_data[label][0]);
        Tensor *in_tensor = new WrappedRamTensor<float>({1, 32, 32, 3}, data);
        get_cifar10_cnn_ctx(ctx, in_tensor);
        S_TENSOR logits = ctx.get("fully_connect_2/logits:0");
        ctx.eval();
        size_t pred_label = argmax(logits);
        bool is_correct = false;
        if (pred_label == label)
        {
            acc += 1.0 / num_imgs;
            is_correct = true;
        }
        printf("pred label: %lu, expecting %lu%s\n", pred_label, label, is_correct ? "" : " (miss)");
    }
    printf("accuracy: %0.2f%%\n", acc * 100);
    return 0;
}
