"""
A script to run experiments over parla app
"""
import re
import subprocess
import argparse
import os

# experiment setting
ITERATION = 5  # run how many times to collect median
GPU_RANGE = ["0,1,2,3", "0,1", "0"]  # run with 1, 2, 4 gpus
branch_list = ["dev"]
KEY_WORD = "time"  # lowercase

# format: > python3 run_app.py <cmd_to_run>
parser = argparse.ArgumentParser()
parser.add_argument("cmd", type=str)
args = parser.parse_args()

def main():
    for branch in branch_list:
        #try:
        #    subprocess.run(["git", "checkout", branch])
        #except:
        #    print(f"[run_app.py]: Check out to branch {branch} failed.")
        #    return

        subprocess.run(args.cmd, shell=True)
        print("[run_app.py]: Warm up run finished.")

        for gpu_setting in GPU_RANGE:
            time_eplased_list = []
            for i in range(ITERATION):
                try:
                    print(f"\n[run_app.py]: Run {i + 1} on gpu {gpu_setting} and branch {branch}")
                    result = subprocess.check_output(
                        args.cmd,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_setting),
                        stderr=subprocess.STDOUT,
                        shell=True
                        )
                except subprocess.CalledProcessError:
                    print("[run_app.py]: CalledProcessError")
                    print("Output:")
                    print(result)
                    return

                # check output for numbers
                last_line = None
                result = result.decode()
                for line in result.splitlines():
                    if KEY_WORD in line.casefold():
                        last_line = line

                if last_line is not None:
                    print(f"\n[run_app.py]: Captured time info: [{last_line}]")
                    time_eplased = re.findall("\d+\.\d+", last_line)
                    if len(time_eplased) != 0:
                        time_eplased_list.append(float(time_eplased[0]))
                else:
                    print(f"[run_app.py]: Failed to find time result in output")
                    return


            if len(time_eplased_list) == ITERATION:
                time_eplased_list.sort()
                print(f"[run_app.py]: Median over {ITERATION} runs is : {time_eplased_list[ITERATION//2]}")
            else:
                print("[run_app.py]: Failed to find median")
                return



if __name__=="__main__":
    main()

