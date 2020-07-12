/***********************************************
	streamcluster.cpp
	: original source code of streamcluster with minor
		modification regarding function calls
	
	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by
	
	Sang-Ha (a.k.a Shawn) Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science
	
***********************************************/

#include "cuda_runtime_api.h"
#include "unistd.h"
#include <string.h>
#include <memory>

#include "streamcluster_header.hpp"
#include "interface.h"

using namespace std;

#define MAXNAMESIZE 1024 			// max filename length
#define SEED 1
#define SP 1 						// number of repetitions of speedy must be >=1
#define ITER 3 						// iterate ITER* k log k times; ITER >= 1
//#define PRINTINFO 				// Enables printing output
#define PROFILE 					// Enables timing info
//#define ENABLE_THREADS			// Enables parallel execution
//#define INSERT_WASTE				// Enables waste computation in dist function
#define CACHE_LINE 512				// cache line in byte


void inttofile(int data, char *filename){

  FILE *fp = fopen(filename, "w");
  fprintf(fp, "%d ", data);
  fclose(fp);
}

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

int isIdentical(float *i, float *j, int D){
// tells whether two points of D dimensions are identical

  int a = 0;
  int equal = 1;

  while (equal && a < D) {
    if (i[a] != j[a]) equal = 0;
    else a++;
  }
  if (equal) return 1;
  else return 0;

}

/* comparator for floating point numbers */
static int floatcomp(const void *i, const void *j)
{
  float a, b;
  a = *(float *)(i);
  b = *(float *)(j);
  if (a > b) return (1);
  if (a < b) return (-1);
  return(0);
}

/* shuffle points into random order */
void shuffle(Points *points)
{
  long i, j;
  Point temp;
  for (i=0;i<points->num-1;i++) {
    j=(lrand48()%(points->num - i)) + i;
    temp = points->p[i];
    points->p[i] = points->p[j];
    points->p[j] = temp;
  }
}

/* shuffle an array of integers */
void intshuffle(std::vector<int> intarray, int length)
{
  long i, j;
  int temp;
  for (i=0;i<length;i++) {
    j=(lrand48()%(length - i))+i;
    temp = intarray[i];
    intarray[i]=intarray[j];
    intarray[j]=temp;
  }
}

#ifdef INSERT_WASTE
float waste(float s )
{
  for( int i =0 ; i< 4; i++ ) {
    s += pow(s,0.78);
  }
  return s;
}
#endif

/* compute Euclidean distance squared between two points */
float dist(Point p1, Point p2, int dim)
{
  int i;
  float result=0.0;
  for (i=0;i<dim;i++)
    result += (p1.coord[i] - p2.coord[i])*(p1.coord[i] - p2.coord[i]);
#ifdef INSERT_WASTE
  float s = waste(result);
  result += s;
  result -= s;
#endif
  return(result);
}


/* run speedy on the points, return total cost of solution */
float pspeedy(Points *points, float z, long & kcenter)
{
  //my block
  long k1 = 0;
  long k2 = points->num;

  static float totalcost;

  /* create center at first point, send it to itself */
  for( int k = 0; k < points->num; k++ )    {
    float distance = dist(points->p[k],points->p[0],points->dim);
    points->p[k].cost = distance * points->p[k].weight;
    points->p[k].assign=0;
  }

  kcenter = 1;

  for(int i = 1; i < points->num; i++ )  {
    bool to_open = ((float)lrand48()/(float)INT_MAX)<(points->p[i].cost/z);
    if( to_open )  {
      kcenter++;

      for( int k = 0; k < points->num; k++ )  {
        float distance = dist(points->p[i], points->p[k], points->dim);
        if( distance * points->p[k].weight < points->p[k].cost )  {
          points->p[k].cost = distance * points->p[k].weight;
          points->p[k].assign=i;
        }
      }
    }
  }

  float mytotal = 0;
  for( int k = k1; k < k2; k++ )  {
    mytotal += points->p[k].cost;
  }

  // aggregate costs from each thread
  totalcost = z * kcenter + mytotal;

  return totalcost;
}


/* facility location on the points using local search */
/* z is the facility cost, returns the total cost and # of centers */
/* assumes we are seeded with a reasonable solution */
/* cost should represent this solution's cost */
/* halt if there is < e improvement after iter calls to gain */
/* feasible is an array of numfeasible points which may be centers */
float pFL(Points *points, state_t & state, args_t args,
          float z, float cost, float e)
{
  /* continue until we run iter iterations without improvement */
  /* stop instead if improvement is less than e */

//  while (change/cost > 1.0*e) {
  for (int j = 0; j < 1 ; j++) {
    float change = 0.0;

    /* randomize order in which centers are considered */
    intshuffle(state.feasible, state.numFeasible);

//    for (i=0;i<iter;i++) {
    for ( int i = 0; i < 1; i++) {
      long x = i % state.numFeasible;

      // Main CUDA function
      change += pgain(state.feasible[x], points, state, args, z);
    }

    cost -= change;
  }

  return(cost);
}

int selectfeasible_fast(Points *points, std::vector<int> & feasible, int kmin)
{
  int numfeasible = points->num;
  if (numfeasible > (ITER*kmin*log((float)kmin))) {
    numfeasible = (int)(ITER*kmin*log((float)kmin));
  }

  feasible.clear();
  feasible.resize(numfeasible, 0);

  float totalweight;

  /*
     Calcuate my block.
     For now this routine does not seem to be the bottleneck, so it is not parallelized.
     When necessary, this can be parallelized by setting k1 and k2 to
     proper values and calling this routine from all threads ( it is called only
     by thread 0 for now ).
     Note that when parallelized, the randomization might not be the same and it might
     not be difficult to measure the parallel speed-up for the whole program.
   */
  //  long bsize = numfeasible;
  long k1 = 0;
  long k2 = numfeasible;

  float w;
  int l,r,k;

  /* not many points, all will be feasible */
  if (numfeasible == points->num) {
    for (int i=k1;i<k2;i++) {
      feasible[i] = i;
    }

    return numfeasible;
  }

  float accumweight[points->num];
  accumweight[0] = points->p[0].weight;
  totalweight = 0;

  for( int i = 1; i < points->num; i++ ) {
    accumweight[i] = accumweight[i-1] + points->p[i].weight;
  }

  totalweight = accumweight[points->num-1];

  for(int i=k1; i<k2; i++ ) {
    w = (lrand48()/(float)INT_MAX)*totalweight;
    //binary search
    l=0;
    r=points->num-1;
    if( accumweight[0] > w )  {
      feasible[i]=0;
      continue;
    }
    while( l+1 < r ) {
      k = (l+r)/2;
      if( accumweight[k] > w ) {
        r = k;
      }
      else {
        l=k;
      }
    }
    feasible[i] = r;
  }

  return numfeasible;
}

/* compute approximate kmedian on the points */
float pkmedian(Points *points, args_t & args, state_t & state) {
  float cost;
  float lastcost;

  float hiz = 0.0;
  float loz = 0.0;

  long k1 = 0;
  long k2 = points->num;

  for (long kk = 0; kk < points->num; kk++ ) {
    hiz += dist(points->p[kk], points->p[0], points->dim)
             * points->p[kk].weight;
  }

  float z = (hiz + loz) / 2.0;

  /* NEW: Check whether more centers than points! */
  if (points->num <= args.kmax) {
    /* just return all points as facilities */
    for (long kk = 0; kk < points->num; kk++) {
      points->p[kk].assign = kk;
      points->p[kk].cost = 0;
    }

    state.kfinal = state.k;
    return 0;
  }

  // Call pspeedy...
  shuffle(points);
  cost = pspeedy(points, z, state.k);

  /* give speedy SP chances to get at least kmin/2 facilities */
  if (state.k < args.kmin) {
    cost = pspeedy(points, z, state.k);
  }

  /* if still not enough facilities, assume z is too high */
  while (state.k < args.kmin) {
    hiz = z;
    z = (hiz+loz)/2.0;
    shuffle(points);
    cost = pspeedy(points, z, state.k);
  }

  /* now we begin the binary search for real */
  /* must designate some points as feasible centers */
  /* this creates more consistancy between FL runs */
  /* helps to guarantee correct # of centers at the end */
  state.numFeasible = selectfeasible_fast(points, state.feasible, args.kmin);

  for( int i = 0; i< points->num; i++ ) {
    state.is_center[points->p[i].assign]= true;
  }

//  while(1) {
  for (int i = 0; i < 1; i++) {

    /* first get a rough estimate on the FL solution */
    //    pthread_barrier_wait(barrier);
    lastcost = cost;
    cost = pFL(points, state, args, z, cost, 0.1);

    /* if number of centers seems good, try a more accurate FL */
    if (((state.k <= (1.1) * args.kmax)&&(state.k >= (0.9) * args.kmin))||
        ((state.k <= args.kmax+2)&&(state.k >= args.kmin-2))) {

      /* may need to run a little longer here before halting without
	 improvement */

      cost = pFL(points, state, args, z, cost, 0.001);
    }

    if (state.k > args.kmax) {
      /* facilities too cheap */
      /* increase facility cost and up the cost accordingly */
      loz = z;
      z = (hiz+loz)/2.0;
      cost += (z-loz)*state.k;
    }

    if (state.k < args.kmin) {
      /* facilities too expensive */
      /* decrease facility cost and reduce the cost accordingly */
      hiz = z;
      z = (hiz+loz)/2.0;
      cost += (z-hiz)*state.k;
    }

    /* if k is good, return the result */
    /* if we're stuck, just give up and return what we have */
    if (((state.k <= args.kmax)&&(state.k >= args.kmin))||((loz >= (0.999)*hiz)) )
    {
      break;
    }
  }

  state.kfinal = state.k;

  return cost;
}

/* compute the means for the k clusters */
int contcenters(Points *points)
{
  long i, ii;
  float relweight;

  for (i=0;i<points->num;i++) {
    /* compute relative weight of this point to the cluster */
    if (points->p[i].assign != i) {
      relweight=points->p[points->p[i].assign].weight + points->p[i].weight;
      relweight = points->p[i].weight/relweight;
      for (ii=0;ii<points->dim;ii++) {
        points->p[points->p[i].assign].coord[ii]*=1.0-relweight;
        points->p[points->p[i].assign].coord[ii]+=
            points->p[i].coord[ii]*relweight;
      }
      points->p[points->p[i].assign].weight += points->p[i].weight;
    }
  }

  return 0;
}

/* copy centers from points to centers */
void copycenters(Points *points, Points* centers, long* centerIDs, long offset)
{
  bool *is_a_median = (bool *) calloc(points->num, sizeof(bool));

  /* mark the centers */
  for ( long i = 0; i < points->num; i++ ) {
    is_a_median[points->p[i].assign] = 1;
  }

  long k=centers->num;

  /* count how many  */
  for ( long i = 0; i < points->num; i++ ) {
    if ( is_a_median[i] ) {
      memcpy( centers->p[k].coord, points->p[i].coord, points->dim * sizeof(float));
      centers->p[k].weight = points->p[i].weight;
      centerIDs[k] = i + offset;
      k++;
    }
  }

  centers->num = k;

  free(is_a_median);
}

void outcenterIDs( Points* centers, long* centerIDs, const char* outfile ) {
  FILE* fp = fopen(outfile, "w");
  if( fp==NULL ) {
    fprintf(stderr, "error opening %s\n",outfile);
    exit(1);
  }
  int* is_a_median = (int*)calloc( sizeof(int), centers->num );
  for( int i =0 ; i< centers->num; i++ ) {
    is_a_median[centers->p[i].assign] = 1;
  }

  for( int i = 0; i < centers->num; i++ ) {
    if( is_a_median[i] ) {
      fprintf(fp, "%ld\n", centerIDs[i]);
      fprintf(fp, "%lf\n", centers->p[i].weight);
      for( int k = 0; k < centers->dim; k++ ) {
        fprintf(fp, "%lf ", centers->p[i].coord[k]);
      }
      fprintf(fp,"\n\n");
    }
  }
  fclose(fp);
}

void localSearch(Points* points, args_t & args, state_t & state) {
  // Serina: strip away multi-threaded implementations
  pkmedian(points, args, state);
}

void streamCluster(std::unique_ptr<PStream> & stream,
                   args_t & args,
                   state_t & state) {
  auto block =
      std::unique_ptr<float[]>(new float[args.chunksize * args.dim]);
  auto centerBlock =
      std::unique_ptr<float[]>(new float[args.clustersize * args.dim]);
  auto centerIDs =
      std::unique_ptr<long[]>(new long[args.clustersize * args.dim]);

  // Init points
  Points points;
  points.dim = args.dim;
  points.num = args.chunksize;
  points.p = std::unique_ptr<Point[]>(new Point[args.chunksize]);

  for( int i = 0; i < args.chunksize; i++ ) {
    points.p[i].coord = &(block[i * args.dim]);
  }

  // Init centers
  Points centers;
  centers.dim = args.dim;
  centers.num = 0;
  centers.p = std::unique_ptr<Point[]>(new Point[args.clustersize]);

  for( int i = 0; i < args.clustersize; i++ ) {
    centers.p[i].coord = &(centerBlock[i * args.dim]);
    centers.p[i].weight = 1.0;
  }

  long IDoffset = 0;

  while(!(stream->feof())) {
    size_t numRead  = stream->read(block, args.dim, args.chunksize );
    fprintf(stderr,"read %lu points\n",numRead);

    if (stream->ferror() || (numRead < args.chunksize && !stream->feof())) {
      fprintf(stderr, "error reading data!\n");
      exit(1);
    }

    points.num = numRead;
    for( int i = 0; i < points.num; i++ ) {
      points.p[i].weight = 1.0;
    }

    state.switch_membership = std::unique_ptr<bool[]>(new bool[points.num]);
    state.is_center.resize(points.num, false);
    state.center_table = std::unique_ptr<int[]>(new int[points.num]);

    localSearch(&points, args, state);

    fprintf(stderr,"finish local search\n");

    contcenters(&points);

    state.isCoordChanged = true;

    if( state.kfinal + centers.num > args.clustersize ) {
      //here we don't handle the situation where # of centers gets too large.
      fprintf(stderr,"oops! no more space for centers\n");
      exit(1);
    }

    copycenters(&points, &centers, centerIDs.get(), IDoffset);
    IDoffset += numRead;
  }

  //finally cluster all temp centers
  state.switch_membership = std::unique_ptr<bool[]>(new bool[centers.num]);
  state.is_center.resize(centers.num, false);
  state.center_table = std::unique_ptr<int[]>(new int[centers.num]);

  localSearch(&centers, args, state);
  contcenters(&centers);
  outcenterIDs(&centers, centerIDs.get(), args.outfileName.c_str());
}

void print_usage(char* appName) {
  fprintf(stderr,
          "usage: %s k1 k2 d n chunksize clustersize infile outfile nproc\n",
          appName);
  fprintf(stderr,"  k1:          Min. number of centers allowed\n");
  fprintf(stderr,"  k2:          Max. number of centers allowed\n");
  fprintf(stderr,"  d:           Dimension of each data point\n");
  fprintf(stderr,"  n:           Number of data points\n");
  fprintf(stderr,"  chunksize:   Number of data points to handle per step\n");
  fprintf(stderr,"  clustersize: Maximum number of intermediate centers\n");
  fprintf(stderr,"  infile:      Input file (if n<=0)\n");
  fprintf(stderr,"  outfile:     Output file\n");
  fprintf(stderr,"  nproc:       Number of threads to use\n");
  fprintf(stderr,"\n");
  fprintf(stderr,
          "if n > 0, points will be randomly generated "
          "instead of reading from infile.\n");
  exit(1);
}

int main_streamcluster(int argc, char **argv, int uid, cudaStream_t & cuda_stream)
{
  // Sanity check on argument number
  if (argc < 10) {
    print_usage(argv[0]);
  }

  std::cout << "Launching streamcluster (modified version by Serina)."
            << std::endl;

  // Parse input arguments
  args_t args = args_t(argc, argv, uid, cuda_stream);

  // State of application that will be updated throughout execution
  state_t state = {};

  // Seed random generator
  srand48(SEED);

  std::unique_ptr<PStream> stream;
  if( args.n > 0 ) {
    stream = std::unique_ptr<PStream>(new SimStream(args.n));
  } else {
    stream = std::unique_ptr<PStream>(new FileStream(args.infileName.c_str()));
  }

  // Main execution
  streamCluster(stream, args, state);

  freeDevMem(state);

  return 0;
}
