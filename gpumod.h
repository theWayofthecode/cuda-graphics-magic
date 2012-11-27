#ifndef _GPUMOD_H
#define _GPUMOD_H

#ifndef _MAGICK_EFFECT_H

typedef unsigned char Quantum;

typedef struct _PixelPacket
{
    Quantum
        red,
        green,
        blue,
        opacity;
} PixelPacket;

#endif

#ifdef __cplusplus
extern "C"
#endif
void median_gpu(PixelPacket *p, PixelPacket *q,  unsigned long columns, unsigned long rows, int radius);


#ifdef __cplusplus
extern "C"
#endif
void convolve_gpu(PixelPacket *p, PixelPacket *q, unsigned long columns,
        unsigned long rows, double *kernel, unsigned int order);
#endif
