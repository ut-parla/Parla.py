'''

Parla: hello.py

Sample Output:
Hello, World!

This is a Parla tutorial introducing the Parla runtime, where a simple console-printing task is defined and executed with the Parla context. Here, we import the Parla context to be used and the "spawn" decorator, the main mechanism for instantiating Parla tasks.  We also define the main function, where the entirety of the Parla exectution is housed, and we define the hello_world() function within the main function.  This task's sole responsibility is to print "Hello, World!" to the console.

'''

# Import Parla
from parla import Parla, spawn
# Import the 'cpu' device type
from parla.cpu import cpu


# Define the main function (recommended when using Parla)
# Tasks cannot be defined in the global scope
def main():

    # Spawn a task to be scheduled by the Parla runtime
    @spawn()
    def hello_world():
        print("Hello, World!", flush=True)

# Execute the Parla program within the Parla context
if __name__ == "__main__":
    with Parla():
        main()
