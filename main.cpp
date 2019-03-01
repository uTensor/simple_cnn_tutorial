#include "models/cifar10_cnn.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include <stdio.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>
#include <mbed.h>
#include <string>

using namespace std;

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char *argv[])
{
    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    TensorIdxImporter t_import;
    char buff[25];
    for (int label = 0; label < 10; ++label)
    {
        Context ctx;
        sprintf(buff, "/fs/idx_data/%i.idx", label);
        string img_path(buff);
        printf("processing: %s\n", buff);
        Tensor *in_tensor = t_import.float_import(img_path);
        printf("image loaded\n");
        get_cifar10_cnn_ctx(ctx, in_tensor);
        S_TENSOR pred = ctx.get("pred:0");
        ctx.eval();
        int pred_label = *(pred->read<int>(0, 0));
        printf("\n");
        printf("pred label: %i, expecting %i\n",
               pred_label,
               label);
    }

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
    return 0;
}
