from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace


async def main():
    task = TaskSpace("Task")
    for i in range(5):
        @spawn(task[i])
        def t2():
            print("Before Task[", i, "]", flush=True)

    await task

    for i in range(5, 10):
        @spawn(task[i])
        def t2():
            print("After Task[", i, "]", flush=True)

    await task

    for i in range(3):
        @spawn(task[i], task[i-1])
        def worker():
            i = i + 4
            print("Capture Check: ", i, flush=True)


if __name__ == "__main__":
    with Parla():
        await main()
