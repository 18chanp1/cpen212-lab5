#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "classifier.h"

inline double max4(double x1, double x2, double x3, double x4) {
    double x = (x2 > x1) ? x2 : x1;
    x = (x3 > x) ? x3 : x;
    x = (x4 > x) ? x4 : x;
    return x;
}

void conv1(const double *wts, const double *bias, const double *iacts, double *oacts, size_t batch_sz) {
    for (size_t x = 0; x < IN2_X; ++x) {
        for (size_t y = 0; y < IN2_Y; ++y) {
            for (size_t k = 0; k < CONV1_K; ++k) {
                for (size_t n = 0; n < batch_sz; ++n) {
                    oacts[(n * CONV1_K * IN2_X * IN2_Y) + (k * IN2_X * IN2_Y) + (y * IN2_X) + x] = bias[k];
                }
            }
        }
    }
    for (size_t r = 0; r < CONV1_R; ++r) {
        for (size_t s = 0; s < CONV1_S; ++s) {
            for (size_t x = 0; x < IN2_X; ++x) {
                for (size_t y = 0; y < IN2_Y; ++y) {
                    for (size_t c = 0; c < CONV1_C; ++c) {
                        for (size_t k = 0; k < CONV1_K; ++k) {
                            for (size_t n = 0; n < batch_sz; ++n) {
                                oacts[(n * CONV1_K * IN2_X * IN2_Y) + (k * IN2_X * IN2_Y) + (y * IN2_X) + x] +=
                                        (wts[(k * CONV1_C * CONV1_S * CONV1_R) + (c * CONV1_S * CONV1_R) + (s * CONV1_R) + r] * iacts[(n * CONV1_C * IN1_X * IN1_Y) + (c * IN1_X * IN1_Y) + ((y + s) * IN1_X) + (x + r)]);
                            }
                        }
                    }
                }
            }
        }
    }
    for (size_t x = 0; x < IN2_X; ++x) {
        for (size_t y = 0; y < IN2_Y; ++y) {
            for (size_t k = 0; k < CONV1_K; ++k) {
                for (size_t n = 0; n < batch_sz; ++n) {
                    if (oacts[(n * CONV1_K * IN2_X * IN2_Y) + (k * IN2_X * IN2_Y) + (y * IN2_X) + x] < 0)
                        oacts[(n * CONV1_K * IN2_X * IN2_Y) + (k * IN2_X * IN2_Y) + (y * IN2_X) + x] = 0;
                }
            }
        }
    }
}

void conv2(const double *wts, const double *bias, const double *iacts, double *oacts, size_t batch_sz) {
    for (size_t k = 0; k < CONV2_K; ++k) {
        for (size_t x = 0; x < IN3_X; ++x) {
            for (size_t y = 0; y < IN3_Y; ++y) {
                for (size_t n = 0; n < batch_sz; ++n) {
                    oacts[(n * CONV2_K * IN3_X * IN3_Y) + (k * IN3_X * IN3_Y) + (y * IN3_X) + x] = bias[k];
                }
            }
        }
    }
    for(int i30y = 0; i30y < IN3_Y; i30y+=4){
        for(int i30x = 0; i30x < IN3_X; i30x += 4){
            for (size_t n = 0; n < batch_sz; ++n) {
                for (size_t k = 0; k < CONV2_K; ++k) {
                    for (size_t c = 0; c < CONV2_C; ++c) {
                        // int i30y = 0;
                        for (size_t y = i30y; y < IN3_Y && y < i30y + 4; ++y) {
                            for (size_t x = i30x; x < IN3_X && x < i30x + 12; ++x) {
                                for (size_t s = 0; s < CONV2_S; ++s) {
                                    for (size_t r = 0; r < CONV2_R; ++r) {
                                        oacts[(n * CONV2_K * IN3_X * IN3_Y) + (k * IN3_X * IN3_Y) + (y * IN3_X) + x] +=
                                                (wts[(k * CONV2_C * CONV2_S * CONV2_R) + (c * CONV2_S * CONV2_R) + (s * CONV2_R) + r] * iacts[(n * CONV2_C * IN2_X * IN2_Y) + (c * IN2_X * IN2_Y) + ((y + s) * IN2_X) + (x + r)]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
    
        }
     }


}

void maxpool(double *iacts, double *oacts, size_t batch_sz) {
    for (size_t c = 0; c < IN4_C; ++c) {
        for (size_t x = 0; x < IN4_X; ++x) {
            for (size_t y = 0; y < IN4_Y; ++y) {
                for (size_t n = 0; n < batch_sz; ++n) {
                    oacts[(n * IN4_C * IN4_X * IN4_Y) + (c * IN4_X * IN4_Y) + (y * IN4_X) + x] =
                            max4(iacts[(n * IN3_C * IN3_X * IN3_Y) + (c * IN3_X * IN3_Y) + ((y + y + 0) * IN3_X) + (x + x + 0)],
                                 iacts[(n * IN3_C * IN3_X * IN3_Y) + (c * IN3_X * IN3_Y) + ((y + y + 0) * IN3_X) + (x + x + 1)],
                                 iacts[(n * IN3_C * IN3_X * IN3_Y) + (c * IN3_X * IN3_Y) + ((y + y + 1) * IN3_X) + (x + x + 0)],
                                 iacts[(n * IN3_C * IN3_X * IN3_Y) + (c * IN3_X * IN3_Y) + ((y + y + 1) * IN3_X) + (x + x + 1)]);
                }
            }
        }
    }
}

void fc1(const double *wts, const double *bias, const double *iacts, double *oacts, size_t batch_sz) {
    for (size_t k = 0; k < FC1_K; ++k) {
        for (size_t n = 0; n < batch_sz; ++n) {
            oacts[(n * FC1_K) + k] = bias[k];
        }
    }
    for(size_t n0 = 0; n0 < batch_sz; n0 += 128){
        for(size_t c0 = 0; c0 < FC1_C; c0 += 128){
            for(size_t k0 = 0; k0 < FC1_K; k0 += 128){
                for (size_t n = n0; n < n0 + 128 && n < batch_sz; ++n) {
                    for (size_t k = k0; k < k0 + 128 && k < FC1_K; ++k) {
                        for (size_t c = c0; c < c0 + 128 && c < FC1_C; ++c){
                            oacts[(n * FC1_K) + k] += wts[k * FC1_C + c] * iacts[(n * FC1_C) + c];
                        }
                    }
                }
            }
        }
    }
    for (size_t k = 0; k < FC1_K; ++k) {
        for (size_t n = 0; n < batch_sz; ++n) {
            if (oacts[(n * FC1_K) + k] < 0)
                oacts[(n * FC1_K) + k] = 0;
        }
    }
}

void fc2(const double *wts, const double *bias, const double *iacts, double *oacts, size_t batch_sz) {
    for (size_t k = 0; k < FC2_K; ++k) {
        for (size_t n = 0; n < batch_sz; ++n) {
            oacts[(n * FC2_K) + k] = bias[k];
        }
    }
    for (size_t c = 0; c < FC2_C; ++c) {
        for (size_t k = 0; k < FC2_K; ++k) {
            for (size_t n = 0; n < batch_sz; ++n) {
                oacts[(n * FC2_K) + k] += wts[k * FC2_C + c] * iacts[(n * FC2_C) + c];
            }
        }
    }
}


//  register volatile pee = wts[(k * CONV2_C * CONV2_S * CONV2_R) + (c * CONV2_S * CONV2_R) + (s * CONV2_R) + r];
//  register volatile que = oacts[(n * CONV2_K * IN3_X * IN3_Y) + (k * IN3_X * IN3_Y) + (y * IN3_X) + x];
//  register volatile arr = iacts[(n * CONV2_C * IN2_X * IN2_Y) + (c * IN2_X * IN2_Y) + ((y + s) * IN2_X) + (x + r)];