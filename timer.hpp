#pragma once

#include <string>
#include <iostream>
#include <map>
#include <chrono>

// This class takes a time stamp when created
// and when deleted, and stores internally a
// list of labels with corresponding accumulated
// time and call counts. At the end of a program,
// you can call the static function summarize() to
// get s a list of timing results for different sections
// in the code.
//
// Example:
//
// void foo()
// {
//    Timer timer("foo");
//    (do some work)
//  }
//
// int main()
//{
//  Timer("main");
//  ...
// foo();
// Timer::summarize();
//}
//
// Note: you can add scopes inside the code to to time code sections as you wish.
//
class Timer
{
public:

  Timer(std::string label);
  ~Timer();
  static void summarize(std::ostream& os=std::cout);

private:

  std::string label_;
  double t_start_;
  static std::map<std::string, double> times_;
  static std::map<std::string, int> counts_;

};
