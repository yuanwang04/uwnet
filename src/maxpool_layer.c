#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


float get_data(matrix m, int w, int h, int c, int idx, int width, int height, int channel) {
    // printf("w: %d, h: %d, c: %d, idx: %d, width: %d, height: %d, channel: %d\n", w, h, c, idx, width, height, channel);
    if (w < 0 || w >= width || h < 0 || h >= height) {
        return 0;
    }
    int pos = idx*width*height*channel + c*width*height + h*width + w;
    // printf("get_data pos: %d\n", pos);
    return m.data[pos];
}


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (int i = 0; i < out.rows; i++) {  // each image
        for (int c = 0; c < l.channels; c++) {  // each channel
            for (int h = 0; h < outh; h++) {
                for (int w = 0; w < outw; w++) {
                    // find center
                    int inw = w * l.stride;
                    int inh = h * l.stride;
                    // find max
                    float max = get_data(in, inw, inh, c, i, l.width, l.height, l.channels);
                    for (int dh = 0; dh < l.size; dh++) {
                        for (int dw = 0; dw < l.size; dw++) {
                            int currW = inw + dw - (l.size - 1) / 2;
                            int currH = inh + dh - (l.size - 1) / 2;
                            max = MAX(max, get_data(in, currW, currH, c, i, l.width, l.height, l.channels));
                        }
                    }
                    int idx = i*outw*outh*l.channels + c*outh*outw + h*outw + w;
                    out.data[idx] = max;
                }
            }
        }
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    for (int i = 0; i < dx.rows; i++) {  // each image
        for (int c = 0; c < l.channels; c++) {  // each channel
            for (int h = 0; h < outh; h++) {
                for (int w = 0; w < outw; w++) {
                    // find center
                    int inw = w * l.stride;
                    int inh = h * l.stride;
                    // find max
                    int maxW = inw;
                    int maxH = inh;
                    float max = get_data(in, inw, inh, c, i, l.width, l.height, l.channels);
                    for (int dh = 0; dh < l.size; dh++) {
                        for (int dw = 0; dw < l.size; dw++) {
                            int currW = inw + dw - (l.size - 1) / 2;
                            int currH = inh + dh - (l.size - 1) / 2;
                            float val = get_data(in, currW, currH, c, i, l.width, l.height, l.channels);
                            if (val > max) {
                                max = val;
                                maxW = currW;
                                maxH = currH;
                            }
                        }
                    }
                    if (maxW < 0 || maxW >= l.width || maxH < 0 || maxH >= l.height) {
                       printf("max=%.5f, maxH=%d, maxW=%d\n", max, maxH, maxW);
                    }
                    int dy_idx = i*outw*outh*l.channels + c*outh*outw + h*outw + w;
                    float dy_val = dy.data[dy_idx];
                    int dx_idx = i*l.channels*l.height*l.width + c*l.height*l.width + maxH*l.width + maxW;
                    dx.data[dx_idx] += dy_val;
                }
            }
        }
    }


    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

