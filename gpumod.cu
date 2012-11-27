/*
MEDIAN FILTER


Estw aktina r kai eikona eisodou width x height.
Ta nhmata organwnontai se block diastasewn 32x32.
Ta block organwnontai se grid diastasewn (width + 31)/32 x (height + 31)/32.
H pros0esh tou 31 ginetai gia na paroume ton amesws megalutero akeraio,
efoson h diairesh den einai teleia (opote 0a uparxoun kai nhmata pou de 0a kanoun kati).
Ka0e nhma (idx,idy) analambanei na upologisei ena pixel ths eikonas ejodou.

gia na upologisoume th mesaia timh enos didiastatou pinaka nxn,
0a xreiastei na exoume enan akoma boh0htiko pinaka nxn stoixeiwn etsi wste na mhn allajoume
ton arxiko.
Omws, mporoume na thn upologisoume, briskontas prwta th mesaia timh ka0e grammhs
kai meta th mesaia timh autwn twn mesaiwn timwn. Etsi, oi apaithseis se mnhmh einai
2n.
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include "gpumod.h"

#define MAX_RADIUS 20
#define MAX_SIZE  ((MAX_RADIUS)<<1)+1


__device__ void  swap (unsigned char *a, unsigned char *b);
__device__ unsigned char getMedian(unsigned char *a, int n);

__global__ void median (unsigned char * img, int width, int height, int radius, unsigned char *img_out) {


	int r;
	int k,l;
	int i,j;
	unsigned char buf[MAX_SIZE];
	unsigned char middle[MAX_SIZE];
//	unsigned char my_shared_array[2*radius+32][2*radius+32];
	extern __shared__ unsigned char my_shared_array[];
	int s_width;
	int offset, offset_bottom, offset_top;
	int row, col;


	s_width = 2*radius + 32;

	col = blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y * blockDim.y + threadIdx.y;

	my_shared_array[(radius+threadIdx.y)*s_width+radius+threadIdx.x] = img[row*width+col];

	if (threadIdx.y == 0) {
		for (i = 0; i < radius; i++) {
			my_shared_array[i*s_width + radius+threadIdx.x] = -((threadIdx.x & 1) ^ (i & 1)); //arxikopoihsh panw grammwn
			my_shared_array[(i+radius+32)*s_width+radius+threadIdx.x] = -((threadIdx.x & 1) ^ (i & 1)); //katw grammwn
			my_shared_array[(radius+threadIdx.x)*s_width+i] = -((threadIdx.x & 1) ^ (i & 1));
			my_shared_array[(radius+threadIdx.x)*s_width+32+radius+i] = -((threadIdx.x & 1) ^ (i & 1));
		}
	}
	else if (threadIdx.y == 1) {
		for (i = 0; i < radius; i++) {
			if (threadIdx.x < radius) {
				my_shared_array[i*s_width+threadIdx.x] = -((threadIdx.x & 1) ^ (i & 1));
				my_shared_array[i*s_width+threadIdx.x + 32 + radius] = -((threadIdx.x & 1) ^ (i & 1));
				my_shared_array[(i+ radius + 32)*s_width+threadIdx.x] = -((threadIdx.x & 1) ^ (i & 1));
				my_shared_array[(i + radius + 32)*s_width+threadIdx.x + radius + 32] = -((threadIdx.x & 1) ^ (i & 1));
			}
		}
	}

	/*uparxoun geitonika pixel pros oles tis kateu0unseis gia na feroun sth koinh mnhmh*/
	if ((blockIdx.x > 0) && (blockIdx.x < (gridDim.x -1)) && (blockIdx.y>0) && (blockIdx.y<(gridDim.y-1))) {

		if (threadIdx.y == 0) { //to prwto warp analambanei na ferei tis epipleon panw grammes
			for (i = 0; i < radius; i++) {
				my_shared_array[i*s_width+radius+threadIdx.x] = img[(row-threadIdx.y-radius+i)*width+col];
			}
		}
		else if (threadIdx.y==1) { // to deutero warp analambanei na ferei tis teleutaies grammes
			for (i = 0; i < radius; i++)
				my_shared_array[(radius+32+i)*s_width+radius+threadIdx.x] = img[(row - threadIdx.y + blockDim.y + i)*width+col];
		}
		else if (threadIdx.y == 2) { //to trito warp analambanei na ferei tiw aristeres sthles
			offset = (row-threadIdx.y)*width+blockIdx.x*blockDim.x; //arxh tou upopinaka eikonas
			for (i = 0; i < radius; i++)
				my_shared_array[(threadIdx.x+radius)*s_width+i] = img[offset+threadIdx.x*width-radius+i];
		}
		else if (threadIdx.y == 3) { //to tetarto warp analambanei na ferei tis dejies sthles
			offset = (row-threadIdx.y)*width+blockIdx.x*blockDim.x; //arxh
			for (i = 0; i < radius; i++)
				my_shared_array[(radius+threadIdx.x)*s_width+32+radius+i] = img[offset+threadIdx.x*width+32+i];
		}
		else if (threadIdx.y == 4) {
			// gwniaka block
			offset_top = (row-threadIdx.y)*width+blockIdx.x*blockDim.x - radius*width - radius;//deixnei thn arxh tou aristerou upopinaka rxr pou 0a antigrafei
			offset_bottom = (row-threadIdx.y+blockDim.y)*width - radius;
			for (i = 0; i < radius; i++) {
				if (threadIdx.x < radius) {
					my_shared_array[i*s_width+threadIdx.x] = img[offset_top+i*width+threadIdx.x]; // panw aristero
					my_shared_array[i*s_width+32+radius+threadIdx.x] = img[offset_top+i*width+threadIdx.x+(32+radius)]; //panw deji
					my_shared_array[(i+radius+32)*s_width+threadIdx.x] = img[offset_bottom+i*width+threadIdx.x]; //katw aristero
					my_shared_array[(i+radius+32)*s_width+threadIdx.x+radius+32] = img[offset_bottom+i*width+threadIdx.x+(32+radius)]; //katw deji
				}
			}
		}
	}
	__syncthreads();
	r = (radius<<1) + 1;

	k = 0;
	for (i = -radius; i < radius+1; i++) {
		l = 0;
		for (j = -radius; j < radius+1; j++) {
			buf[l++] = my_shared_array[(threadIdx.y + radius +i)*s_width+threadIdx.x+radius+j];
		}
		middle[k++] = getMedian(buf, r);
	}
	img_out[row*width+col] = getMedian(middle, r);
}

///////////////////////////////////////////////
__device__ void  swap (unsigned char *a, unsigned char *b) {

	int c;
	c = *b;
	*b = *a;
	*a = c;
}
/////////////////////////////////////////////////
/* dexetai enan pinaka a, n stoixeiwn kai epistrefei th mesaia timh*/
/*o algori0mos basizetai sto quicksort, alla den ginetai plhrhs tajinomhsh tou a*/

__device__ unsigned char getMedian(unsigned char *a, int n) {

	int l,r,i,j,middle_pos;
	unsigned char p;
	middle_pos = n>>1;

	l = 0;
	r = n-1;
	p = a[0];
	i = l+1;
	j = r;

	while(i < j) {

		while(i <= j) {
			while (a[i] <= p && i < n) i++;
			while (a[j] > p && j >=0) j--;
			swap (&a[i], &a[j]);
		}
		swap(&a[i], &a[j]);
		swap(&a[l], &a[j]);
		if (j == middle_pos) return a[j];
		else if (j > middle_pos) {
			r = j-1;
			i = l+1;
			p = a[l];
			j = r;
		}
		else {
			l = j+1;
			p = a[l];
			i = j+2;
			j = r;
		}
	}
	return a[i];
}
////////////////////////////////////////////
void median_gpu(PixelPacket *p, PixelPacket *q, unsigned long columns, unsigned long rows, int radius)
{
    unsigned char *image_in, *image_out;  //pinakes gia th gpu
    unsigned long npixels = columns * rows;
    unsigned char *img = (unsigned char *)malloc(npixels);
    cudaError_t memcpy_a_err;
    int i;
    struct timeval start_time, stop_time, start_timewithout, stop_timewithout;

    cudaSetDevice(0);

    gettimeofday(&start_time, NULL);

    for (i = 0; i < npixels; i++) {
        img[i] = p[i].blue;
    }

    cudaMalloc((void**) &image_in, npixels);
    cudaMalloc((void**) &image_out, npixels);

    if (image_in == NULL || image_out == NULL) {
		fprintf(stderr, "could not malloc\n");
		exit(1);
    }
    dim3 gridDim((columns+31)>>5, (rows+31)>>5);

    dim3 blockDim(32,32);

    memcpy_a_err = cudaMemcpy(image_in, img, npixels, cudaMemcpyHostToDevice);
    if (memcpy_a_err) {
		fprintf(stderr, "could not transfer from host to device\n");
		exit(1);
    }
  //  median_old<<<gridDim, blockDim>>>(image_in, columns, rows, radius, image_out);

    median<<<gridDim, blockDim, (2*radius+32)*(2*radius+32)>>>(image_in, columns, rows, radius, image_out);
    memcpy_a_err = cudaMemcpy(img, image_out, npixels, cudaMemcpyDeviceToHost);
    gettimeofday(&stop_time, NULL);
    printf("Time: %lu\n", (stop_time.tv_sec-start_time.tv_sec)*1000000 + (stop_time.tv_usec - start_time.tv_usec));


    if (memcpy_a_err) {
        fprintf(stderr, "could not transfer from device to host\n");
        exit(1);
    }

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
    // something's gone wrong
    // print out the CUDA error as a string
    	fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));

    // we can't recover from the error -- exit the program
    	return;
    }

    for (i = 0; i < npixels; i++) {
        q[i].blue = img[i];
    }
}

//********************************************************************
//                    convolution                                    *
//********************************************************************

__global__ void _convolve(PixelPacket *img, float *kernel, unsigned int order)
{
    unsigned int idx, idy, i = 0, j;
    unsigned int radius = order >> 1;
    unsigned int radiusX2 = radius << 1;
    extern __shared__ PixelPacket sharray[];
    float pixel = 0.0;
    float *lker = (float *)&sharray[0];
    PixelPacket *limg = &sharray[order * order];
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    idy = blockIdx.y * blockDim.y + threadIdx.y;
    //global to share
    if (threadIdx.x < order && threadIdx.y < order) {
        i = threadIdx.x * order + threadIdx.y;
        lker[i] = kernel[i];
    }
    limg[threadIdx.x * (32 + radiusX2) + threadIdx.y] =\
    img[idx * gridDim.y * blockDim.y + idy];
    if (threadIdx.x < radiusX2) {  //possible optimization for idle threads
        limg[(threadIdx.x + 32) * (32 + radiusX2) + threadIdx.y] =\
        img[(idx + 32) * gridDim.y * blockDim.y + idy];
    }
    if (threadIdx.y < radiusX2) {
        limg[threadIdx.x * (32 + radiusX2) + threadIdx.y + 32] =\
        img[idx * gridDim.y * blockDim.y + idy + 32];
    }
    if (threadIdx.x < radiusX2 && threadIdx.y < radiusX2) {
        limg[(threadIdx.x + 32) * (32 + radiusX2) + threadIdx.y + 32] =\
        img[(idx + 32) * gridDim.y * blockDim.y + idy + 32];
    }
    __threadfence_block();
    //compute
    for (i = 0; i < order; i++) {
        for (j = 0; j < order; j++) {
            pixel += kernel[i * order + j] *\
                    limg[(threadIdx.x + i) * (radiusX2 + 32) +\
                    threadIdx.y + j].blue;
        }
    }
    __threadfence();
    img[(idx + radius) * gridDim.y * blockDim.y + idy + radius].blue =\
    RoundFloatToQuantum(pixel);
}

void convolve_gpu(PixelPacket *p, PixelPacket *q, unsigned long columns,
        unsigned long rows, double *kernel, unsigned int order)
{
    unsigned int i, off;
    unsigned int radius = order >> 1;
    float *d_kernel, *tmp;
    PixelPacket *d_img;
    //init memory
    tmp = (float *)malloc(sizeof(float) * order * order);
    cudaMalloc((void **)&d_kernel, sizeof(float) * order * order);
    cudaMalloc((void **)&d_img, sizeof(PixelPacket) * (rows + (radius << 1)) *\
                                (columns + (radius << 1)));
    assert(d_kernel != NULL && d_img != NULL);
    for (i = 0; i < order * order; i++)
        tmp[i] = (float)kernel[i];
    cudaMemcpy(d_kernel, tmp, sizeof(float) * order * order, cudaMemcpyHostToDevice);
    cudaMemset((void **)d_img, 0, sizeof(PixelPacket) * (rows + (radius << 1)) * (columns + (radius << 1)));
    for (i = 0; i < rows; i++) {
        off = (radius + i) * (columns + (radius << 1)) + radius;
        cudaMemcpy(d_img + off, &p[i * columns], sizeof(PixelPacket) * columns,
                cudaMemcpyHostToDevice);
    }
    assert(columns * rows > 1024);
    //invoke kernel
    dim3 gridDim((rows + 16) >> 5, (columns + 16) >> 5);
    dim3 blockDim(32, 32);
    _convolve<<<gridDim, blockDim, 4 * ((order * order) +\
    (32 + (radius << 1)) * (32 + (radius << 1)))>>>(d_img, d_kernel, order);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(error));
        exit(2);
    }
    //write result
    for (i = 0; i < rows; i++) {
        off = (radius + i) * (columns + (radius << 1)) + radius;
        cudaMemcpy(&q[i * columns], d_img + off, sizeof(PixelPacket) * columns,
                cudaMemcpyDeviceToHost);
    }
    cudaFree(d_kernel);
    cudaFree(d_img);
}
