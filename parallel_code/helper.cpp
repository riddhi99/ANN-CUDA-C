#include <random>
#include "helper.h"

std::default_random_engine& Random_Gen()
{
  static std::default_random_engine ran;
  return ran;
};
