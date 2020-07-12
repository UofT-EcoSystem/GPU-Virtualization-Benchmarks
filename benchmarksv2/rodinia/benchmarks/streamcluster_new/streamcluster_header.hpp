/************************************************
	streamcluster_cuda_header.cu
	: header file to streamcluster
	
	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by
	
	Sang-Ha (a.k.a Shawn) Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science
	
***********************************************/

#ifndef STREAMCLUSTER_CUDA_HEADER_CU
#define STREAMCLUSTER_CUDA_HEADER_CU

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include <stdlib.h>
#include <sys/time.h>
#include <string>
#include <assert.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>

#include <cuda.h>

using namespace std;

struct args_t {
  long kmin;
  long kmax;
  int dim;
  long n;
  long chunksize;
  long clustersize;
  string infileName;
  string outfileName;
  int nproc;

  // Framework params
  cudaStream_t cuda_stream;
  int uid;

  args_t(int argc, char **argv, int uid, cudaStream_t & cuda_stream) {
    this->uid = uid;
    this->cuda_stream = cuda_stream;

    this->kmin = atoi(argv[1]);
    this->kmax = atoi(argv[2]);
    this->dim = atoi(argv[3]);
    this->n = atoi(argv[4]);
    this->chunksize = atoi(argv[5]);
    this->clustersize = atoi(argv[6]);

    this->infileName = std::string(argv[7]);
    this->outfileName = std::string(argv[8]);

    this->nproc = atoi(argv[9]);
  }
};


/* this structure represents a point */
/* these will be passed around to avoid copying coordinates */
struct Point {
  float weight;
  float *coord;
  long assign;  /* number of point where this one is assigned */
  float cost;  /* cost of that assignment, weight*distance */
};

/* this is the array of points */
struct Points{
  long num; /* number of points; may not be N if this is a sample */
  int dim;  /* dimensionality */
//  std::vector<Point> p;
  std::unique_ptr<Point[]> p; /* the array itself */
};

struct state_t {
  //whether to switch membership in pgain
  std::unique_ptr<bool[]> switch_membership;

  //whether a point is a center
  std::vector<bool> is_center;

  //index table of centers
//  std::vector<int> center_table;
  std::unique_ptr<int[]> center_table;

  bool isCoordChanged = false;
  long kfinal;

  long k;
  std::vector<int> feasible;
  int numFeasible;

  int iter = 0;
  std::unique_ptr<float[]> coord_h;

  float *coord_d;
  int   *center_table_d;
  bool  *switch_membership_d;
  Point *p;
};

class PStream {
public:
  virtual size_t read( std::unique_ptr<float[]> & dest, int dim, int num ) = 0;
  virtual int ferror() = 0;
  virtual int feof() = 0;
  virtual ~PStream() {
  }
};

//synthetic stream
class SimStream : public PStream {
public:
  SimStream(long n_ ) {
    n = n_;
  }
  size_t read( std::unique_ptr<float[]> & dest, int dim, int num ) {
    size_t count = 0;
    for( int i = 0; i < num && n > 0; i++ ) {
      for( int k = 0; k < dim; k++ ) {
	dest[i*dim + k] = lrand48()/(float)INT_MAX;
      }
      n--;
      count++;
    }
    return count;
  }
  int ferror() {
    return 0;
  }
  int feof() {
    return n <= 0;
  }
  ~SimStream() { 
  }
private:
  long n;
};

class FileStream : public PStream {
public:
  FileStream(const char* filename) {
    fp = fopen( filename, "rb");
    if( fp == NULL ) {
      fprintf(stderr,"error opening file %s\n.",filename);
      exit(1);
    }
  }
  size_t read( std::unique_ptr<float[]> & dest, int dim, int num ) {
    return std::fread(dest.get(), sizeof(float)*dim, num, fp);
  }
  int ferror() {
    return std::ferror(fp);
  }
  int feof() {
    return std::feof(fp);
  }
  ~FileStream() {
    printf("closing file stream\n");
    fclose(fp);
  }
private:
  FILE* fp;
};

/* function prototypes */
int isIdentical(float*, float*, int);
//static int floatcomp(const void*, const void*);
float waste(float);
float pgain_old(long, Points*, float, long int*, int, pthread_barrier_t*);
float pgain(int x, Points* points, state_t & state, args_t & args, float z);

void freeDevMem(state_t & state);
float dist(Point p1, Point p2, int dim);

#endif
