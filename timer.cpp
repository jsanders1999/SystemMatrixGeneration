#include "timer.hpp"

#include <iomanip>

// for timing routine
#include <omp.h>

#include <iostream>
#include <iomanip>
#include <cmath.h>

// static members of a class must be defined
// somewhere in an object file, otherwise you
// will get linker errors (undefined reference)
std::map<std::string, int> Timer::counts_;
std::map<std::string, double> Timer::times_;

  Timer::Timer(std::string label)
  : label_(label)
  {
    t_start_ = omp_get_wtime();
  }


  Timer::~Timer()
  {
    double t_end = omp_get_wtime();
    times_[label_] += t_end - t_start_;
    squared_times_[label_] += (t_end - t_start_)*(t_end - t_start_);
    counts_[label_]++;
  }

void Timer::summarize(std::ostream& os)
{
  os << "==================== TIMER SUMMARY =========================================" << std::endl;
  os << "label                                   \tcalls     \ttotal time\tmean time\tstd time "<<std::endl;
  os << "----------------------------------------------" << std::endl;
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    double sigma = sqrt(squared_times_[label]/count-time*time/double(count*count))
    std::cout << std::setw(40) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) << time/double(count) << "\t" << std::setw(10) << sigma << std::endl;
  }
  os << "============================================================================" << std::endl;
}
