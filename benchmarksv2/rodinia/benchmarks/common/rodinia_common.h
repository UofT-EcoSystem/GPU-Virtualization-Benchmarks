#ifndef RODINIA_COMMON_H
#define RODINIA_COMMON_H

long long get_time();
void fatal(const char *s);


// Rodinia Hotspot constant params
/* chip parameters  */
const float t_chip = 0.0005;
const float chip_height = 0.016;
const float chip_width = 0.016;
/* ambient temperature, assuming no package at all  */
const float amb_temp = 80.0;


#endif
