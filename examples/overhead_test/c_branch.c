#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include<unistd.h>
#include <omp.h>
#include<iostream>
#include<chrono>
#include<time.h>

using namespace std;

void busy_sleep(const unsigned milli){
    auto block = chrono::milliseconds(milli);
    auto time_start = chrono::high_resolution_clock::now();
    while(chrono::duration<double>(chrono::high_resolution_clock::now()-time_start) < block){
    }
}

void branch(int depth, int threshold)
{
  double start = omp_get_wtime();
  busy_sleep(1000);
  double end = omp_get_wtime();
  std::cout<<depth<<" "<<end-start<<std::endl;

  if (depth+1>threshold) return;
#pragma omp task untied
  branch(depth+1,threshold);
#pragma omp task untied
  branch(depth+1,threshold);
}


int main(int argc, char *argv[])
{
  int max_depth = 2;
  double start,stop;
  start = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp single nowait
    branch(0,max_depth);
#pragma omp taskwait
  }
  stop = omp_get_wtime();

  printf("Elapsed time %lf\n",(stop-start));

  return 0;
}

