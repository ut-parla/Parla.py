from parla import Parla
from parla.cpu import cpu
from parla.tasks import *

def main():
    @spawn()
    def hello_world():
        print("Hello, World!")


if __name__ == "__main__":
    with Parla():
        main()
