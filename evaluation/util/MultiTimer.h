#ifndef HECO_BENCH_MULTITIMER_H
#define HECO_BENCH_MULTITIMER_H

#include <chrono>
#include <iostream>
#include <vector>

#define INIT_TIME_IT() int __timer_index
#define TIME_IT(TIMER, CODE)            \
    __timer_index = TIMER.startTimer(); \
    CODE TIMER.stopTimer(__timer_index)

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePoint;
typedef std::chrono::microseconds TimeDuration;

class MultiTimer
{
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
    void printToStream(std::ostream &out);
    void printToFile(const std::string &filename);
};

#endif // HECO_BENCH_MULTITIMER_H
