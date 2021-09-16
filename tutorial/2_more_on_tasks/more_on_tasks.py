from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn, TaskSpace


def main():
    @spawn(placement=cpu)
    async def task_launcher():
        task = TaskSpace("Task")
        for i in range(5):
            @spawn(task[i])
            def t2():
                print("Before Task[", i, "]", flush=True)

        await task
        print('---------')
        for i in range(5, 10):
            @spawn(task[i])
            def t2():
                print("After Task[", i, "]", flush=True)

        await task
        print('---------')

        task2 = TaskSpace("Task2")

        for i in range(3):
            dep = [task2[i-1]] if (i)>0 else []
            @spawn(task2[i], dep)
            def t2():
                nonlocal i
                i = i + 4
                print("Check Task[", i, "]", flush=True)
            #t2()



if __name__ == "__main__":
    with Parla():
        main()
