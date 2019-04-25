

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
//#include "prepsubband_gpu.h"
//#include "prepsubband_cmd.h"
#include "stddef.h"

#include "device_functions.h"

#include "stdio.h"
#include <limits.h>
#include <ctype.h>


#include "chkio.h"
//#include "makeinf.h"
#define SUBSBLOCKLEN 1024
#define BLOCKSTOSKIP 100
#define SWAP(a,b) tempzz=(a);(a)=(b);(b)=tempzz;

//#include "median.c"
#include "misc_utils.h"

//#include "prepsubband_cmd.h"
#include "mask.h"
//#include "multibeam.h"
//#include "bpp.h"
//#include "wapp.h"
//#include "gmrt.h"
//#include "spigot.h"
//#include "psrfits.h"
#include "psrfits.h"
/*
#include "psrfits.c"
#include "vectors.c"
#include "cal2mjd.c"
#include "misc_utils.c"
#include "cldj.c"
#include "dispersion.c"
#include "mask.c"
#include "transpose.c"
#include "median.c"
#include "clipping.c"
*/

/*
 __host__ __device__ int read_PSRFITS_subbands(float *data, double *dispdelays, int numsubbands,
                          int transpose, int *padding,
                          int *maskchans, int *nummasked, mask * obsmask);
*/
static __global__ void float_dedisp_kernel(float *d_data, float *d_lastdata,
                                    int numpts, int numchan,
                                    int **d_delays, float approx_mean, float **d_result);

extern "C" 
int get_data_gpu(int downsamp, int numdms, int nsub, FILE * infiles[], int numfiles, float **outdata,
                 int numchan, int blocklen, int blocksperread,
                 mask * obsmask, float *padvals, double dt,
                 double *dispdts, int **offsets, int *padding, short **subsdata);



 
int *gen_ivect(long length)
{
   int *v;

   v = (int *) fftwf_malloc((size_t) (sizeof(int) * length));
   if (!v) {
      perror("\nError in gen_ivect()");
      printf("\n");
      exit(-1);
   }
   return v;
}

float *gen_fvect(long length)
{
   float *v;

   v = (float *) fftwf_malloc((size_t) (sizeof(float) * length));
   if (!v) {
      perror("\nError in gen_fvect()");
      printf("\n");
      exit(-1);
   }
   return v;
}

void vect_free( void *vect )
{
   fftwf_free(vect);
}





/*
static int read_subbands(FILE * infiles[], int numfiles,
                         float *subbanddata, double timeperblk,
                         int *maskchans, int *nummasked, mask * obsmask,
                         float clip_sigma, float *padvals)
// Read short int subband data written by prepsubband 
{
   int ii, jj, index, numread = 0, mask = 0,offset;
   short subsdata[SUBSBLOCKLEN]; 
   double starttime, run_avg;
   float subband_sum;
   static int currentblock = 0;

   if (obsmask->numchan) mask = 1;
   
   // Read the data 
   for (ii = 0; ii < numfiles; ii++) {
      numread = chkfread(subsdata, sizeof(short), SUBSBLOCKLEN, infiles[ii]);
      run_avg = 0.0;
      if (cmd->runavgP==1) {
          for (jj = 0; jj < numread ; jj++)
              run_avg += (float) subsdata[jj];
          run_avg /= numread;
      }
      for (jj = 0, index = ii; jj < numread; jj++, index += numfiles)
         subbanddata[index] = (float) subsdata[jj] - run_avg;
      for (jj = numread; jj < SUBSBLOCKLEN; jj++, index += numfiles)
         subbanddata[index] = 0.0;
      index += numread;
   }

   if (mask) {
      starttime = currentblock * timeperblk;
      *nummasked = check_mask(starttime, timeperblk, obsmask, maskchans);
   }

   // Clip nasty RFI if requested and we're not masking all the channels
   if ((clip_sigma > 0.0) && !(mask && (*nummasked == -1))){
      subs_clip_times(subbanddata, SUBSBLOCKLEN, numfiles, clip_sigma, padvals);
   }

   // Mask it if required 
   if (mask && numread) {
      if (*nummasked == -1) {  
	 // If all channels are masked 
         for (ii = 0; ii < SUBSBLOCKLEN; ii++)
            memcpy(subbanddata + ii * numfiles, padvals, sizeof(float) * numfiles);
      } else if (*nummasked > 0) {     
 	// Only some of the channels are masked 
         int offset, channum;
         for (ii = 0; ii < SUBSBLOCKLEN; ii++) {
            offset = ii * numfiles;
            for (jj = 0; jj < *nummasked; jj++) {
               channum = maskchans[jj];
               subbanddata[offset + channum] = padvals[channum];
            }
         }
      }
   }

   // Zero-DM removal if required 
   if (cmd->zerodmP==1) {
       for (ii = 0; ii < SUBSBLOCKLEN; ii++) {
           offset = ii * numfiles;
           subband_sum = 0.0; 
           for (jj = offset; jj < offset+numfiles; jj++) {
               subband_sum += subbanddata[jj];
           }    
           subband_sum /= (float) numfiles;
           // Remove the channel average 
           for (jj = offset; jj < offset+numfiles; jj++) {
               subbanddata[jj] -= subband_sum;
           }    
       }    
   }

   currentblock += 1;
   return numread;
}

*/



static __global__ void float_dedisp_kernel(float *d_data, float *d_lastdata,
                                    int numpts, int numchan,
                                    int **d_delays, float approx_mean, float **d_result)
{

  int b_idx = blockIdx.x;
  int b_idy = blockIdx.y;
  int b_idz = blockIdx.z;
  int t_idx = threadIdx.x;
  int s = b_idx;
  int i = b_idy;
  int k = b_idz;
  for (int x = t_idx; x < blockDim.x * gridDim.z; x += blockDim.x)
    d_result[s][x] = -approx_mean;
  __syncthreads();

  int j;
  int t = d_delays[s][i];

  if (k < numpts - t) {
    j = i + d_delays[s][i] * numchan + k * numchan;
    d_result[s][k] += d_lastdata[j];
  } else {
    j = i + k * numchan;
    d_result[s][k] += d_data[j];
  }
}


int get_data_gpu(int downsamp, int numdms, int nsub, FILE * infiles[], int numfiles, float **outdata,
                 int numchan, int blocklen, int blocksperread,
                 mask * obsmask, float *padvals, double dt,
                 double *dispdts, int **offsets, int *padding, short **subsdata)
{
  int firsttime = 1, *maskchans = NULL, blocksize;

  static int worklen, dsworklen;
  static float *tempzz, *data1, *data2, *dsdata1 = NULL, *dsdata2 = NULL;
  static float *currentdata, *lastdata, *currentdsdata, *lastdsdata;
  static double blockdt;
  int totnumread = 0, numread = 0, ii, jj, tmppad = 0, nummasked = 0;

  if (firsttime) {

    //if (cmd->maskfileP)
if (1)
      maskchans = gen_ivect(numchan);
//	maskchans = (int *)malloc(numchan*sizeof(int));

    worklen = blocklen * blocksperread;
    //dsworklen = worklen / cmd->downsamp;
dsworklen = worklen / downsamp;
    { // Make sure that out working blocks are long enough...
      for (ii = 0; ii < numchan; ii++) {
        if (dispdts[ii] > worklen)
          printf("WARNING!:  (dispdts[%d] = %.0f) > (worklen = %d)\n",
                 ii, dispdts[ii], worklen);
      }

      for (ii = 0; ii < numdms; ii++) {
        for (jj = 0; jj < nsub; jj++) {
          if (offsets[ii][jj] > dsworklen)
            printf("WARNING!:  (offsets[%d][%d] = %d) > (dsworklen = %d)\n",
                   ii, jj, offsets[ii][jj], dsworklen);
        }
      }
    }

    blocksize = blocklen * nsub;
    blockdt = blocklen * dt;
    data1 = gen_fvect(nsub * worklen);
    data2 = gen_fvect(nsub * worklen);
    currentdata = data1;
    lastdata = data2;
    if (downsamp > 1) {
      dsdata1 = gen_fvect(nsub * dsworklen);
      dsdata2 = gen_fvect(nsub * dsworklen);
      currentdsdata = dsdata1;
      lastdsdata = dsdata2;
    } else {
      currentdsdata = data1;
      lastdsdata = data2;
    }
  }
  while (1) {
    if (1) {
      for (ii = 0; ii < blocksperread; ii++) {
/*
        if (cmd->pkmbP)
          numread = read_PKMB_subbands(infiles, numfiles,
                                       currentdata + ii * blocksize,
                                       dispdts, cmd->nsub, 0, &tmppad,
                                       maskchans, &nummasked, obsmask);
        else if (cmd->bcpmP)
          numread = read_BPP_subbands(infiles, numfiles,
                                      currentdata + ii * blocksize,
                                      dispdts, cmd->nsub, 0, &tmppad,
                                      maskchans, &nummasked, obsmask, ifs);
        else if (cmd->spigotP)
          numread = read_SPIGOT_subbands(infiles, numfiles,
                                         currentdata + ii * blocksize,
                                         dispdts, cmd->nsub, 0,
                                         &tmppad, maskchans,
                                         &nummasked, obsmask, ifs);
*/
        //else if (cmd->psrfitsP)

	  if(1)
          numread = read_PSRFITS_subbands(currentdata + ii * blocksize,
                                          dispdts, nsub, 0,
                                          &tmppad, maskchans,
                                          &nummasked, obsmask);


/*
        else if (cmd->wappP)
          numread = read_WAPP_subbands(infiles, numfiles,
                                       currentdata + ii * blocksize,
                                       dispdts, cmd->nsub, 0, &tmppad,
                                       maskchans, &nummasked, obsmask, ifs);
        else if (cmd->gmrtP)
          numread = read_GMRT_subbands(infiles, numfiles,
                                       currentdata + ii * blocksize,
                                       dispdts, cmd->nsub, 0, &tmppad,
                                       maskchans, &nummasked, obsmask);
        else if (cmd->filterbankP)
          numread = read_filterbank_subbands(infiles, numfiles,
                                             currentdata +
                                             ii * blocksize, dispdts,
                                             cmd->nsub, 0, &tmppad,
                                             maskchans, &nummasked, obsmask);
       
	 else if (insubs)
          numread = read_subbands(infiles, numfiles,
                                  currentdata + ii * blocksize,
                                  blockdt, maskchans, &nummasked,
                                  obsmask, cmd->clip, padvals);
*/     
	 if (!firsttime)
          totnumread += numread;
        if (numread != blocklen) {
          for (jj = ii * blocksize; jj < (ii + 1) * blocksize; jj++)
            currentdata[jj] = 0.0;
        }
        if (tmppad)
          *padding = 1;
      }
    }
    /* Downsample the subband data if needed */
    if (downsamp > 1) {
      int kk, offset, dsoffset, index, dsindex;
      float ftmp;
      for (ii = 0; ii < dsworklen; ii++) {
        dsoffset = ii * nsub;
        offset = dsoffset * downsamp;
        for (jj = 0; jj < nsub; jj++) {
          dsindex = dsoffset + jj;
          index = offset + jj;
          currentdsdata[dsindex] = 0.0;
          for (kk = 0, ftmp = 0.0; kk < downsamp; kk++) {
            ftmp += currentdata[index];
            index += nsub;
          }
          /* Keep the short ints from overflowing */
          currentdsdata[dsindex] += ftmp / downsamp;
        }
      }
    }
    if (firsttime) {
      SWAP(currentdata, lastdata);
      SWAP(currentdsdata, lastdsdata);
      firsttime = 0;
    } else
      break;
  }

  //////////////////////////////////////////////
  float *d_data, *d_lastdata, **d_outdata;
  int **d_offsets;
  int dataSize = sizeof(float) * nsub * dsworklen;

  int offset_N = numdms;
  int offset_M = nsub;
  int outdata_N = numdms;
  int outdata_M = worklen / downsamp;
  size_t pitch;
  checkCudaErrors(cudaMalloc((void **)&d_data, dataSize));
  checkCudaErrors(cudaMalloc((void **)&d_lastdata, dataSize));
  checkCudaErrors(cudaMallocPitch(&d_offsets, &pitch, offset_M * sizeof(int), offset_N));
  checkCudaErrors(cudaMallocPitch(&d_outdata, &pitch, outdata_M * sizeof(float), outdata_N));

  checkCudaErrors(cudaMemcpy(d_data, currentdsdata, dataSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_lastdata, lastdsdata, dataSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(d_offsets, offset_M * sizeof(float), offsets, pitch, offset_M * sizeof(float), offset_N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(d_outdata, outdata_M * sizeof(float), outdata, pitch, outdata_M * sizeof(float), outdata_N, cudaMemcpyHostToDevice));

  if (!0) {
    /*
    for (ii = 0; ii < cmd->numdms; ii++)
     float_dedisp(currentdsdata, lastdsdata, dsworklen,
      cmd->nsub, offsets[ii], 0.0, outdata[ii]);
    */
    const int BLOCKSIZE = 1024;
    dim3 gridSize(numdms, numchan, dsworklen / BLOCKSIZE);
    dim3 blockSize(BLOCKSIZE);
printf("\n\n\n\nnumdms =%d , numchan =%d , dsworklen/1024 = %d\n\n\n",numdms,numchan,dsworklen/1024);
    float_dedisp_kernel <<< gridSize, blockSize>>>(d_data, d_lastdata, dsworklen, nsub, d_offsets, 0.0, d_outdata);

  checkCudaErrors(cudaMemcpy(currentdsdata, d_data, dataSize, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(lastdsdata, d_lastdata, dataSize, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(offsets, offset_M * sizeof(float), d_offsets, pitch, offset_M * sizeof(float), offset_N, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy2D(outdata, outdata_M * sizeof(float), d_outdata, pitch, outdata_M * sizeof(float), outdata_N, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_lastdata));
  checkCudaErrors(cudaFree(d_offsets));
  checkCudaErrors(cudaFree(d_outdata));


  } else {
    /* Input format is sub1[0], sub2[0], sub3[0], ..., sub1[1], sub2[1], sub3[1], ... */
    float infloat;
    for (ii = 0; ii < nsub; ii++) {
      for (jj = 0; jj < dsworklen; jj++) {
        infloat = lastdsdata[ii + (nsub * jj)];
        subsdata[ii][jj] = (short) (infloat + 0.5);
        //if ((float) subsdata[ii][jj] != infloat)
        //   printf
        //       ("Warning:  We are incorrectly converting subband data! float = %f  short = %d\n",
        //         infloat, subsdata[ii][jj]);
      }
    }
  }
  SWAP(currentdata, lastdata);
  SWAP(currentdsdata, lastdsdata);
  if (totnumread != worklen) {
    if (1)
      vect_free(maskchans);
    vect_free(data1);
    vect_free(data2);
    if (downsamp > 1) {
      vect_free(dsdata1);
      vect_free(dsdata2);
    }
  }
  return totnumread;
}

