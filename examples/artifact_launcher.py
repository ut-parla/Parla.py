import os
import sys
import argparse
import pexpect as pe

######
# Helper functions
######

def parse_times(output):
    times = []
    for line in output.splitlines():
        line = str(line).strip('\'')
        if "Time" in line:
            times.append(float(line.split()[-1].strip()))
    return times





#######
# Define functions to gather each result (per figure, per app)
#######

#Test:
def run_test(gpu_list, timeout):
    output_dict = {}

    #Loop over number of GPUs in each subtest
    for n_gpus in gpu_list:
        command = f"python test_script.py -ngpus {n_gpus}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        assert(output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        output_dict[n_gpus] = times

    return output_dict

#Figure 10: Cholesky
def run_cholesky_28(gpu_list, timeout):
    output_dict = {}

    sub_dict = {}

    print("\t   [Running Cholesky 28K (2Kx2K Blocks) 1/3] Manual Movement, User Placement")
    #Test 1: Manual Movement, User Placement
    for n_gpus in gpu_list:
        command = f"python examples/cholesky/blocked_cholesky_manual.py -ngpus {n_gpus} -fixed 1 -b 2000 -nblocks 14"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        assert(output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times

    output_dict["m,u"] = sub_dict

    #Test 2: Automatic Movement, User Placement
    print("\t   [Running Cholesky 28K (2Kx2K Blocks) 2/3] Automatic Movement, User Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/cholesky/blocked_cholesky_automatic.py -ngpus {n_gpus} -fixed 1 -b 2000 -nblocks 14"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        assert(output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,u"] = sub_dict

    #Test 3: Automatic Movement, Policy Placement
    print("\t   [Running Cholesky 28K (2Kx2K Blocks) 3/3] Automatic Movement, Policy Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/cholesky/blocked_cholesky_automatic.py -ngpus {n_gpus} -fixed 0 -b 2000 -nblocks 14"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        assert(output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict

    return output_dict

#Figure 13: Parla Cholesky (CPU)
def run_cholesky_20_host():
    pass

#Figure 13: Parla Cholesky (GPU)
def run_cholesky_20_gpu(gpu_list):
    pass

#Figure 13: Dask Cholesky (CPU)
def run_dask_cholesky_20_host(cores_list):
    pass

#Figure 13: Dask Cholesky (GPU)
def run_dask_cholesky_20_gpu(gpu_list):
    pass

#Figure 10: Jacobi
def run_jacobi(gpu_list):
    pass


#Figure 10: Matmul
def run_matmul(gpu_list, timeout):

    output_dict = {}

    sub_dict = {}

    print("\t   [Running 1/3] Manual Movement, User Placement")
    #Test 1: Manual Movement, User Placement
    for n_gpus in gpu_list:
        command = f"python examples/matmul/matmul_manual.py -ngpus {n_gpus} -fixed 1 -n 32000"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        assert(output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times

    output_dict["m,u"] = sub_dict

    #Test 2: Automatic Movement, User Placement
    print("\t   [Running 2/3] Automatic Movement, User Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/matmul/matmul_automatic.py -ngpus {n_gpus} -fixed 1 -n 32000"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        assert(output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,u"] = sub_dict

    #Test 3: Automatic Movement, Policy Placement
    print("\t   [Running 3/3] Automatic Movement, Policy Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/matmul/matmul_automatic.py -ngpus {n_gpus} -fixed 0 -n 32000"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        assert(output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict

    return output_dict

#Figure 10: BLR
def run_blr(gpu_list):
    pass

#Figure 10: NBody
def run_nbody(gpu_list):
    pass

#Figure 10: Synthetic Reduction
def run_reduction(gpu_list):
    pass

#Figure 10: Synthetic Independent
def run_independent(gpu_list):
    pass

#Figure 10: Synthetic Serial
def run_serial():
    pass

#Figure 15: Batched Cholesky Variants
def run_batched_cholesky():
    pass

#Figure 12: Prefetching Plot
def run_prefetching_test():
    pass

#Figure 14: GIL test
def run_GIL_test():
    pass

test = [run_test]
#figure_10 = [run_cholesky_28, run_jacobi, run_matmul, run_blr, run_nbody, run_reduction, run_independent, run_serial]
figure_10 = [run_matmul, run_blr, run_nbody, run_reduction, run_independent, run_serial]
figure_13 = [run_cholesky_20_host, run_cholesky_20_gpu, run_dask_cholesky_20_host, run_dask_cholesky_20_gpu]
figure_15 = [run_batched_cholesky]
figure_12 = [run_prefetching_test]

figure_dictionary = {}
figure_dictionary['Figure_10'] = figure_10
figure_dictionary['Figure_13'] = figure_13
figure_dictionary['Figure_15'] = figure_15
figure_dictionary['Figure_12'] = figure_12
figure_dictionary['Figure_test'] = test

figure_output = {}

if __name__ == '__main__':
    import os
    import sys
    parser = argparse.ArgumentParser(description='Runs the benchmarks')
    parser.add_argument('--figures', type=int, nargs="+", help='Figure numbers to generate', default=None)
    parser.add_argument('--timeout', type=int, help='Max Timeout for a benchmark', default=1000)

    args = parser.parse_args()

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is None:
        print("Warning CUDA_VISIBLE_DEVICES is not set.")
        cuda_visible_devices = list(range(4))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        print(f"Setting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)

    ngpus = [1, 2, 4]


    if args.figures is None:
        figure_list = ["-1"]
    else:
        figure_list = args.figures

    for figure_num in figure_list:
        if int(figure_num) < 0:
            figure_num = "test"
        figure = f"Figure_{figure_num}"
        if figure not in figure_dictionary:
            print(f"Experiments for {figure} not found")
            continue

        total_tests = len(figure_dictionary[figure])
        i = 1
        print("Starting collection for :", figure)
        for test in figure_dictionary[figure]:
            test_output = {}
            print("\t ++Experiment {}/{}. Name: {}".format(i, total_tests, test.__name__))
            output_dict = test(ngpus, args.timeout)
            test_output[test.__name__] = output_dict
            print("\t --Experiment {}/{} Completed. Output: {}".format(i, total_tests, output_dict))
            i += 1
        figure_output[figure] = test_output
        print(f"Collection for {figure} complete")


    print("All experiments complete.")
    print("Output:")
    print(figure_output)







