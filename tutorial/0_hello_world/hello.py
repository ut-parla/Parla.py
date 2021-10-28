'''

Parla: hello.py

Sample Output:
Hello, World!

This is a Parla tutorial introducing the Parla runtime, where a simple console-printing task is defined and executed with the Parla context. Here, we import the Parla context to be used and the "spawn" decorator, the main mechanism for instantiating Parla tasks.  We also define the main function, where the entirety of the Parla exectution is housed, and we define the hello_world() function within the main function.  This task's sole responsibility is to print "Hello, World!" to the console.

'''

# Import Parla
from parla import Parla

# Import for the spawn decorator
from parla.tasks import spawn

# Define the main function (required of all Parla implementations)
def main():

    # Spawn a task to be scheduled by the Parla runtime
    @spawn()
    def hello_world():

        # Simply print a console message
        print("Hello, World!")

# Execute the Parla program with the Parla Context
if __name__ == "__main__":
    with Parla():
        main()
