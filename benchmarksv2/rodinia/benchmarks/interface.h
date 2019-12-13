#ifndef RODINIA_INTERFACE_H
#define RODINIA_INTERFACE_H

#ifdef RODINIA_MUMMER
int main_mummer(int argc, char** argv, int uid, cudaStream_t & stream);
#endif

#ifdef RODINIA_HEARTWALL
int main_heartwall(int argc, char** argv, int uid, cudaStream_t & stream);
#endif


#endif

