#include "timer.hpp"

#include <iostream>
#include <cmath>
#include <limits>

#include </apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/gsl-2.7-vpcpzpdpgfisizrdupj6lgwydh5iygwy>

double f(double x, void* params) {
  double alpha = *(double*)params;
  double f = std::log(alpha * x) / std::sqrt(x);
  return f;
}

int main() {
  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);

  double result, error;
  double expected = -4.0;
  double alpha = 1.0;

  gsl_function F;
  F.function = &f;
  F.params = &alpha;

  gsl_integration_qags(&F, 0, 1, 0, 1e-7, 1000, w, &result, &error);

  std::cout << "result          = " << result << std::endl;
  std::cout << "exact result    = " << expected << std::endl;
  std::cout << "estimated error = " << error << std::endl;
  std::cout << "actual error    = " << result - expected << std::endl;
  std::cout << "intervals       = " << w->size << std::endl;

  gsl_integration_workspace_free(w);

  return 0;
}



//int main(int argc, char* argv[])
//{
  
//}

