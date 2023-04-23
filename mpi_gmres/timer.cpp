
#include "timer.hpp"

#include <iomanip>

// for timing routine
#include <omp.h>

#include <iostream>
#include <iomanip>


// static members of a class must be defined
// somewhere in an object file, otherwise you
// will get linker errors (undefined reference)
std::map<std::string, int> Timer::counts_;
std::map<std::string, double> Timer::times_;
std::map<std::string, double> Timer::Ic_;
std::map<std::string, double> Timer::Gf_;
std::map<std::string, double> Timer::Gb_;

Timer::Timer(std::string label, int nx, int ny, int nz)
: label_(label), nx_(nx), ny_(ny), nz_(nz)
{
  t_start_ = omp_get_wtime();
}

Timer::~Timer()
{
  double t_end = omp_get_wtime();
  times_[label_] += t_end - t_start_;
  counts_[label_]++;

  if (label_=="init"){ 
    Ic_[label_] = 0.0;
    Gf_[label_] += 0.0;
    Gb_[label_] += 16.0 * nx_ * ny_ * nz_;}
  else if (label_ == "dot"){
    Ic_[label_] = 0.125;
    Gf_[label_] += 2.0 * nx_ * ny_ * nz_;
    Gb_[label_] += 16.0 * nx_ * ny_ * nz_;}
  else if (label_ == "axpby"){
    Ic_[label_] = 0.125;
    Gf_[label_] += 3.0 * nx_ * ny_ * nz_;
    Gb_[label_] += 24.0 * nx_ * ny_ * nz_;}
  else if (label_ == "apply_stencil3d"){
    Ic_[label_] = 13.0/16.0 - (nx_ * ny_ + ny_ * nz_ + nx_ * nz_)/4.0/(nx_ * ny_ * nz_);
    Gf_[label_] += 13.0 * nx_ * ny_ * nz_ - 4.0 * (nx_ * ny_ + ny_ * nz_ + nx_ * nz_);
    Gb_[label_] += 16.0 * nx_ * ny_ * nz_;}
  else {
    Ic_[label_] = 0.0;
    Gf_[label_] = 0.0;
    Gb_[label_] = 0.0;}
}

void Timer::summarize(std::ostream& os)
{
  os << "======================================================================== TIMER SUMMARY =============================================================================" << std::endl;
  os << "label               \tcalls     \ttotal time\tmean time \tcomputational intensity \tfloating point rate (Gflop/s) \tdata bandwidth (Gbyte/s)"<<std::endl;
  os << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
 
    double ic = Ic_[label]; 
    double gf = Gf_[label];
    double gb = Gb_[label];

    std::cout << std::setw(20) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) << time/double(count) << "\t" << std::setw(20) << ic << "\t" << std::setw(30) << gf/1073741824.0/time << "\t" << std::setw(30) << gb/1073741824.0/time  << std::endl;
  }
  os << "=====================================================================================================================================================================" << std::endl;
}

