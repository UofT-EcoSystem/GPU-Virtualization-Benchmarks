#ifndef RODINIA_INTERFACE_H
#define RODINIA_INTERFACE_H

#ifdef RODINIA_MUMMER
int main_mummer(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef RODINIA_HEARTWALL
int main_heartwall(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef RODINIA_HOTSPOT
int main_hotspot(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef RODINIA_STREAMCLUSTER
int main_streamcluster(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef RODINIA_PATHFINDER
int main_pathfinder(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef RODINIA_LAVAMD
int main_lavamd(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef RODINIA_MYOCYTE
int main_myocyte(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#endif

