import threading
import time
from sleep.core import sleep, bsleep


exitFlag = 0

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print("Starting " + self.name)
      print_time(self.name, 5, self.counter)
      print("Exiting " + self.name)

def print_time(threadName, counter, delay):
   while counter:
      if exitFlag:
         threadName.exit()
      t = time.time()
      bsleep(1000)
      t = time.time() - t
      print("%s: %s" % (threadName, t))
      counter -= 1

print("Running Main Thread")
t = time.time()
bsleep(1000)
t = time.time() - t
print("Main Thread Bsleep Time: ", t)


# Create new threads
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 1)
thread3 = myThread(3, "Thread-3", 1)

# Start new Threads
thread1.start()
thread2.start()
#thread3.start()
print("Exiting Main Thread")
