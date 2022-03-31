#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

#include <unistd.h>
#include <stdint.h>

#define IN1_X (28)
#define IN1_Y (28)
#define CONV1_K (32)
#define CONV1_C (1)
#define CONV1_S (3)
#define CONV1_R (3)
#define CONV1_W (CONV1_K*CONV1_C*CONV1_S*CONV1_R)
#define IN2_X (IN1_X-2)
#define IN2_Y (IN1_Y-2)
#define IN2_C (CONV1_K)
#define CONV2_K (64)
#define CONV2_C (IN2_C)
#define CONV2_S (3)
#define CONV2_R (3)
#define CONV2_W (CONV2_K*CONV2_C*CONV2_S*CONV2_R)
#define IN3_X (IN2_X-2)
#define IN3_Y (IN2_Y-2)
#define IN3_C (CONV2_K)
#define IN4_X (IN3_X/2)
#define IN4_Y (IN3_Y/2)
#define IN4_C (IN3_C)
#define FC1_K (128)
#define FC1_C (IN4_X*IN4_Y*IN4_C)
#define FC1_W (FC1_K * FC1_C)
#define FC2_K (10)
#define FC2_C (FC1_K)
#define FC2_W (FC2_K * FC2_C)

void conv1(const double *wts, const double *bias, const double *iacts, double *oacts, size_t batch_sz);

void conv2(const double *wts, const double *bias, const double *iacts, double *oacts, size_t batch_sz);

void maxpool(double *iacts, double *oacts, size_t batch_sz);

void fc1(const double *wts, const double *bias, const double *iacts, double *oacts, size_t batch_sz);

void fc2(const double *wts, const double *bias, const double *iacts, double *oacts, size_t batch_sz);

#endif // __CLASSIFIER_H__