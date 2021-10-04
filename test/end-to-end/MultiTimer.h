#ifndef AST_OPTIMIZER_MULTITIMER_H
#define AST_OPTIMIZER_MULTITIMER_H

#include <chrono>
#include <vector>
#include <iostream>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePoint;
typedef std::chrono::microseconds TimeDuration;

class MultiTimer {
 private:
  std::vector<TimePoint> *startTimes;
  std::vector<TimePoint> *stopTimes;
  int runCounter;

 public:
   MultiTimer();
   ~MultiTimer();
   int startTimer();
   void stopTimer(int id);
   void addIteration();
   TimeDuration getTime(int id) const;
   void printToStream(std::ostream& out);
   void printToFile(const std::string& filename);
};


#endif //AST_OPTIMIZER_MULTITIMER_H
