#ifndef _HELPER_H
#define _HELPER_H 1
#include <random>
#define INDEX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
std::default_random_engine& Random_Gen();
#endif
