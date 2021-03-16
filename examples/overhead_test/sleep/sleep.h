#include<unistd.h>
#include<time.h>
#include<chrono>

using namespace std;
using namespace chrono;

/*
void busy_sleep(const unsigned milli){
    clock_t time_end;
    time_end = clock() + milli * CLOCKS_PER_SEC/1000;
    while(clock() < time_end)
    {
    }
}
*/

void busy_sleep(const unsigned milli){
    auto block = chrono::milliseconds(milli);
    auto time_start = chrono::high_resolution_clock::now();
    while(chrono::duration<double>(chrono::high_resolution_clock::now()-time_start) < block){
    }
}


void sleeper(const unsigned int t){
    sleep(t);
}
