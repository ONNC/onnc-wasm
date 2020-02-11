extern "C" {
    #include "benchmark.h"
}

#include <cstdint>
#include <unordered_map>
#include <string>
#include <cstdio>
#include <sys/time.h>

class Timer {
public:
  Timer() = default;
  virtual ~Timer() = default;

  virtual void clear() {
    IsRecording = false;
    RecTime = 0;
  }

  void start() {
    gettimeofday(&TStart, nullptr);
    IsRecording = true;
  }

  uint64_t stop() {
    if (IsRecording) {
      struct timeval TEnd;
      gettimeofday(&TEnd, nullptr);
      RecTime += (uint64_t)1000000 * (TEnd.tv_sec - TStart.tv_sec) +
                 TEnd.tv_usec - TStart.tv_usec;
      IsRecording = false;
    }
    return RecTime;
  }

private:
  bool IsRecording = false;
  struct timeval TStart;
  uint64_t RecTime = 0;
};

static Timer& getTimer(){
    static Timer timer;
    return timer;
}

void QITC_time_start(){
    getTimer().start();
};
void QITC_time_stop(){
    fprintf(stderr, "[STATS] Inference cost %ld us\n", getTimer().stop());
};
void QITC_time_clear(){
    getTimer().clear();
};
