import os
import sys
import argparse
import pexpect as pe
import re



######
# Helper functions
######


def wassert(output, condition, required=True, verbose=True):
    if condition:
        return True
    else:
        print("\t   FAILURE:", output[1])
        if verbose:
            print("\t   OUTPUT:", output[0])

        if required:
            raise Exception("Assertion Failure.")
        else:
            return False

def parse_times(output):
    times = []
    for line in output.splitlines():
        line = str(line).strip('\'')
        if "Time" in line:
            times.append(float(line.split()[-1].strip()))
    return times

def parse_nbody_times(output):
    times = []
    for line in output.splitlines():
        line = str(line).strip('\'')
        if "end-to-end     ," in line:
            times.append(float(line.split(",")[-1].strip()))
    return times

def parse_blr_times(output):
    times = []
    for line in output.splitlines():
        line = str(line).strip('\'')
        if "total" in line:
            times.append(float(line.split(",")[-1].strip()))
    return times

def parse_cublas_times(output):
    output = str(output)
    prog = re.findall(r"T:[ ]*\d+\.\d+", output)
    result = prog[-1]
    prog = re.findall(r"\d+\.\d+", result)
    result = prog[-1]
    result = result.strip()
    return result

def parse_synthetic_times(output):
    output = str(output)
    prog = re.findall(r"Graph(.*)\\r\\nParla", output)
    times = []
    for result in prog:
        p = re.findall(r"Median = \d+\.\d+", result)
        r = p[-1]
        p = re.findall(r"\d+\.\d+", r)
        r = p[-1]
        r = r.strip()
        times.append(r)
    if len(times) > 1:
        return times
    return times[0]

def parse_magma_times(output):
    output = str(output)
    prog = re.findall(r"\([ ]*\d+\.\d+\)", output)
    result = prog[-1]
    result = result.strip("()").strip()
    return result


#######
# Define functions to gather each result (per figure, per app)
#######

#Test:
def run_test(gpu_list, timeout):
    """
    Sample function to test launcher
    """

    output_dict = {}

    #Loop over number of GPUs in each subtest

    print("\t   [Running Test Script 1/1]")
    for n_gpus in gpu_list:
        command = f"python test_script.py -ngpus {n_gpus}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        output_dict[n_gpus] = times

    return output_dict

#Figure 9: Cholesky 28K Magma
def run_cholesky_magma(gpu_list, timeout):
    """
    Figure 9. 3rd Party Comparison. Run blocked cholesky factorization on gpus with magma.
    Requires artifact/magma built with testing enabled on the path.
    See README.
    """

    #Put testing directory on PATH
    magma_root = os.environ.get("MAGMA_ROOT")
    if not magma_root:
        raise Exception("MAGMA_ROOT not set")
    os.environ["PATH"] = magma_root+"/testing/"+":"+os.environ.get("PATH")

    output_dict = {}

    print("\t   [Running Cholesky 28K (MAGMA) 1/1]")
    #Loop over number of GPUs in each subtest
    for n_gpus in gpu_list:
        command = f"./testing_dpotrf_mgpu --ngpu {n_gpus} -N 28000"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_magma_times(output)
        print(f"\t    {n_gpus} GPUs: {times}")
        output_dict[n_gpus] = times

    return output_dict

#Figure 9: Cholesky
def run_cholesky_28(gpu_list, timeout):
    """
    Figure 9. Parla. Run blocked cholesky factorization on gpus.
    """

    output_dict = {}

    sub_dict = {}

    # Generate input file
    if not os.path.exists("examples/cholesky/chol_28000.npy"):

        print("\t  --Making input matrix...")
        command = f"python examples/cholesky/make_cholesky_input.py -n 28000 -output examples/cholesky/chol_28000.npy"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        print("\t  --Generated input matrix.")


    print("\t   [Running Cholesky 28K (2Kx2K Blocks) 1/3] Manual Movement, User Placement")
    #Test 1: Manual Movement, User Placement
    for n_gpus in gpu_list:
        command = f"python examples/cholesky/blocked_cholesky_manual.py -ngpus {n_gpus} -fixed 1 -b 2000 -nblocks 14 -matrix examples/cholesky/chol_28000.npy"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times

    output_dict["m,u"] = sub_dict

    #Test 2: Automatic Movement, User Placement
    print("\t   [Running Cholesky 28K (2Kx2K Blocks) 2/3] Automatic Movement, User Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/cholesky/blocked_cholesky_automatic.py -ngpus {n_gpus} -fixed 1 -b 2000 -nblocks 14 -matrix examples/cholesky/chol_28000.npy"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
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
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict

    return output_dict

#Figure 12: Parla Cholesky (CPU)
def run_cholesky_20_host(core_list, timeout):
    """
    Figure 12. Parla. Run blocked cholesky on the CPU
    """

    output_dict = {}

    sub_dict = {}

    # Generate input file
    if not os.path.exists("examples/cholesky/chol_20000.npy"):
        print("\t  --Making input matrix...")
        command = f"python examples/cholesky/make_cholesky_input.py -n 20000 -output examples/cholesky/chol_20000.npy"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        print("\t  --Generated input matrix.")

    #NOTE: Number of CPU cores is set manually.
    cpu_cores = [ 1, 2, 4, 8, 16 ]
    print("\t   Running CPU:")
    #Test 1: Manual Movement, User Placement
    for num_cores in cpu_cores:
        command = f"python examples/cholesky/blocked_cholesky_cpu.py -matrix examples/cholesky/chol_20000.npy -b 2000 -workers {num_cores}"
        print(command, "...")
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {num_cores} CPU cores: {times}")
        sub_dict[num_cores] = times
    output_dict["cpu"] = sub_dict
    return output_dict

#Figure 12: Parla Cholesky (GPU)
def run_cholesky_20_gpu(gpu_list, timeout):
    """
    Figure 12. Parla. Run blocked cholesky on the GPUs.
    """

    output_dict = {}

    sub_dict = {}

    # Generate input file
    if not os.path.exists("examples/cholesky/chol_20000.npy"):
        print("\t  --Making input matrix...")
        command = f"python examples/cholesky/make_cholesky_input.py -n 20000 -output examples/cholesky/chol_20000.npy"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        print("\t  --Generated input matrix.")

    gpu_list = [ 1, 2, 3, 4 ]
    print("\t   Running GPU:")
    #Test 1: Manual Movement, User Placement
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/cholesky/blocked_cholesky_manual.py -matrix examples/cholesky/chol_20000.npy -fix 1 -ngpus {n_gpus}"
        #print("Current running:", command)
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["gpu"] = sub_dict
    return output_dict

#Figure 12: Dask Cholesky (CPU)
def run_dask_cholesky_20_host(cores_list, timeout):
    """
    Figure 12. Dask. Run blocked cholesky on the CPU.
    """

    output_dict = {}

    sub_dict = {}

    # Generate input file
    if not os.path.exists("examples/cholesky/chol_20000.npy"):
        print("\t  --Making input matrix...")
        command = f"python examples/cholesky/make_cholesky_input.py -n 20000 -output examples/cholesky/chol_20000.npy"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        print("\t  --Generated input matrix.")

    worker_list = [ 1, 2, 4 ]
    # Per-thread per each worker.
    perthread_list = [ [ 1 ], [ 1 ], [ 1, 2, 4 ] ]
    print("\t   Running Dask CPU:")
    #Test 1: Manual Movement, User Placement
    for wi in range(len(worker_list)):
        n_workers = worker_list[wi]
        for pt in perthread_list[wi]:
            command = f"python examples/cholesky/dask/dask_cpu_cholesky.py -workers {n_workers} -perthread {pt} -matrix examples/cholesky/chol_20000.npy"
            #print(command, "...")
            output = pe.run(command, timeout=timeout, withexitstatus=True)
            #Make sure no errors or timeout were thrown
            wassert(output, output[1] == 0)
            #Parse output
            times = parse_times(output[0])
            n_threads = n_workers * pt
            print(f"\t\t    {n_threads} CPU cores: {times}")
            sub_dict[n_workers] = times
    output_dict["dask-cpu"] = sub_dict
    return output_dict

#Figure 12: Dask Cholesky (GPU)
def run_dask_cholesky_20_gpu(gpu_list, timeout):
    """
    Figure 12. Dask. Run blocked cholesky on the GPUs.
    """

    output_dict = {}

    sub_dict = {}

    # Generate input file
    if not os.path.exists("examples/cholesky/chol_20000.npy"):
        print("\t  --Making input matrix...")
        command = f"python examples/cholesky/make_cholesky_input.py -n 20000 -output examples/cholesky/chol_20000.npy"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        print("\t  --Generated input matrix.")

    gpu_list = [ 1, 2, 3, 4 ]
    print("\t   Running Dask GPU:")
    #Test 1: Manual Movement, User Placement
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        os.environ['UCX_TLS'] = "cuda,cuda_copy,cuda_ipc,tcp"
        command = f"python examples/cholesky/dask/dask_gpu_cholesky.py -matrix examples/cholesky/chol_20000.npy"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        if "No module named" in str(output[0]):
            print("ERROR", output[0])
            assert(False)
        #Make sure no errors or timeout were thrown
        # DASK-GPU makes an asyncio error after the app completes. so ignore it.
        #wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["dask-gpu"] = sub_dict
    return output_dict


#Figure 9: Jacobi
def run_jacobi(gpu_list, timeout):
    """
    Figure 9. Parla. Run Jacobi on the GPUs.
    """

    output_dict = {}

    sub_dict = {}

    print("\t   [Running 1/3] Manual Movement, User Placement")
    #Test 1: Manual Movement, User Placement
    for n_gpus in gpu_list:
        command = f"python examples/jacobi/jacobi_manual.py -ngpus {n_gpus} -fixed 1"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times

    output_dict["m,u"] = sub_dict

    #Test 2: Automatic Movement, User Placement
    print("\t   [Running 2/3] Automatic Movement, User Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/jacobi/jacobi_automatic.py -ngpus {n_gpus} -fixed 1"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,u"] = sub_dict

    #Test 3: Automatic Movement, Policy Placement
    print("\t   [Running 3/3] Automatic Movement, Policy Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/jacobi/jacobi_automatic.py -ngpus {n_gpus} -fixed 0"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict

    return output_dict

#Figure 9: Matmul 32K cuBLAS
def run_matmul_cublas(gpu_list, timeout):
    """
    Figure 9. 3rd Party. Run Matmul on the GPUs.
    Needs artifact/cublasmg/ to be compiled and on the path. See README.
    """

    #Put testing directory on PATH
    cublasmg_root = os.environ.get("CUBLASMG_ROOT")
    cudamg_root = os.environ.get("CUDAMG_ROOT")
    if not cublasmg_root:
        raise Exception("CUBLASMG_ROOT not set")
    if not cudamg_root:
        raise Exception("CUDAMG_ROOT not set")

    os.environ["LD_LIBRARY_PATH"] = cudamg_root+"/lib/"+":"+os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = cublasmg_root+"/lib/"+":"+os.environ.get("LD_LIBRARY_PATH")
    os.environ["PATH"] = cublasmg_root+"/test/"+":"+os.environ.get("PATH")

    output_dict = {}
    #Loop over number of GPUs in each subtest
    print("\t   [Running Matmul 32K (cuBLAS) 1/1]")
    for n_gpus in gpu_list:
        command = f"./{n_gpus}gpuGEMM.exe"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_cublas_times(output)
        print(f"\t    {n_gpus} GPUs: {times}")
        output_dict[n_gpus] = times

    return output_dict

#Figure 9: Matmul
def run_matmul(gpu_list, timeout):
    """
    Figure 9. Parla. Run Matmul on the GPUs.
    """

    output_dict = {}

    sub_dict = {}

    print("\t   [Running 1/3] Manual Movement, User Placement")
    #Test 1: Manual Movement, User Placement
    for n_gpus in gpu_list:
        command = f"python examples/matmul/matmul_manual.py -ngpus {n_gpus} -fixed 1 -n 32000"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
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
        wassert(output, output[1] == 0)
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
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict
    return output_dict


#Figure 9: blr
def run_blr_threads(gpu_list, timeout):
    """
    Figure 9. Non-Parla. Run BLR on the GPUs.
    """

    output_dict = {}

    # Generate input file
    if not os.path.exists("examples/blr/inputs"):
        print("\t  --Making input directory...")
        os.makedirs("examples/blr/inputs")
        print("\t  --Made input directory.")

    if (not os.path.exists("examples/blr/inputs/matrix_10k.npy")) or (not os.path.exists("examples/blr/inputs/vector_10k.npy")):
        print("\t  --Making input matrix_10k...")
        command = f"python examples/blr/app/main.py -mode gen -type mgpu_blr -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        print("\t  --Generated input matrix_10k.")

    # Test 1: Manual Movement, User Placement
    print("\t   [Running 1/1]")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/blr/app/main.py -mode run -type mgpu_blr -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4 -ngpus {n_gpus}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_blr_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times


    return sub_dict


#Figure 9: blr
def run_blr_parla(gpu_list, timeout):
    """
    Figure 9. Parla. Run BLR on the GPUs.
    """

    output_dict = {}

    # Generate input file
    if not os.path.exists("examples/blr/inputs"):
        print("\t  --Making input directory...")
        os.makedirs("examples/blr/inputs")
        print("\t  --Made input directory.")

    if (not os.path.exists("examples/blr/inputs/matrix_10k.npy")) or (not os.path.exists("examples/blr/inputs/vector_10k.npy")):
        print("\t  --Making input matrix_10k...")
        command = f"python examples/blr/app/main.py -mode gen -type parla -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        print("\t  --Generated input matrix_10k.")

    # Test 1: Manual Movement, User Placement
    print("\t   [Running 1/3] Manual Movement, User Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/blr/app/main.py -mode run -type parla -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4 -fixed 1 -movement lazy -ngpus {n_gpus}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_blr_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
        output_dict["m,u"] = sub_dict

    # Test 2: Automatic Movement, User Placement
    print("\t   [Running 2/3] Automatic Movement, User Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/blr/app/main.py -mode run -type parla -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4 -fixed 1 -movement eager -ngpus {n_gpus}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_blr_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
        output_dict["a,u"] = sub_dict

    # Test 3: Automatic Movement, Policy Placement
    print("\t   [Running 3/3] Automatic Movement, Policy Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/blr/app/main.py -mode run -type parla -matrix examples/blr/inputs/matrix_10k.npy -vector examples/blr/inputs/vector_10k.npy -b 2500 -nblocks 4 -fixed 0 -movement eager -ngpus {n_gpus}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_blr_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
        output_dict["a,p"] = sub_dict

    return output_dict


#Figure 9: NBody (Parla)
def run_nbody_parla(gpu_list, timeout):
    """
    Figure 9. NBody (Parla). Run NBody on the GPUs.
    """

    output_dict = {}

    # Generate input file
    if not os.path.exists("examples/nbody/python-bh/input"):
        print("\t  --Making input directory...")
        os.makedirs("examples/nbody/python-bh/input")
        print("\t  --Made input directory.")

    if not os.path.exists("examples/nbody/python-bh/input/n10M.txt"):
        print("\t  --Making input particle file...")
        command = f"python examples/nbody/python-bh/bin/gen_input.py normal 10000000 examples/nbody/python-bh/input/n10M.txt"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        print("\t  --Made input particle file.")

    # Test 1: Manual Movement, User Placement
    print("\t   [Running 1/3] Manual Movement, User Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla{n_gpus}_eager_sched.ini"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_nbody_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times

        output_dict["m,u"] = sub_dict

    # Test 2: Automatic Movement, User Placement
    print("\t   [Running 2/3] Automatic Movement, User Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla{n_gpus}_eager.ini"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_nbody_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,u"] = sub_dict

    # Test 3: Automatic Movement, Policy Placement
    print("\t   [Running 3/3] Automatic Movement, Policy Placement")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/parla{n_gpus}.ini"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_nbody_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict

    return output_dict



#Figure 9: NBody (Python threading, no tasks)
def run_nbody_threads(gpu_list, timeout):
    """
    Figure 9. NBody (Python threading, no tasks). Run NBody on the GPUs.
    """

    output_dict = {}

    # Generate input dir
    if not os.path.exists("examples/nbody/python-bh/input"):
        print("\t  --Making input directory...")
        os.makedirs("examples/nbody/python-bh/input")
        print("\t  --Made input directory.")

    # Generate input file
    if not os.path.exists("examples/nbody/python-bh/input/n10M.txt"):
        command = f"python examples/nbody/python-bh/bin/gen_input.py normal 10000000 examples/nbody/python-bh/input/n10M.txt"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)

    mode_list=["singlegpu", "2gpus", "4gpus"]
    gpu_list = [1, 2, 4]
    print("\t   [Running 1/1] Python Threading")

    idx = 0
    for mode in mode_list:
        n_gpus = gpu_list[idx]
        command = f"python examples/nbody/python-bh/bin/run_2d.py examples/nbody/python-bh/input/n10M.txt 1 1 examples/nbody/python-bh/configs/{mode}.ini"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_nbody_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        output_dict[n_gpus] = times
        idx += 1

    return output_dict

#Figure 9: Synthetic Reduction
def run_reduction(gpu_list, timeout):
    """
    Figure 9. Parla. Synthetic Reduction. Run Reduction on the GPUs.
    """

    output_dict = {}
    sub_dict = {}

    # Generate input file
    if not os.path.exists("examples/synthetic/inputs"):
        print("\t  --Making input directory...")
        os.makedirs("examples/synthetic/inputs")
        print("\t  --Made input directory.")

    reduction_policy_path = "examples/synthetic/inputs/reduction_gpu_policy.txt"
    reduction_user_path = "examples/synthetic/inputs/reduction_gpu_user.txt"
    if not os.path.exists(reduction_policy_path):
        command = f"python examples/synthetic/graphs/generate_reduce_graph.py -overlap 1 "
        command += f"-level 8 -branch 2 -N 6250 -gil_time 0 -weight 16000 "
        command += f"-user 0 -output {reduction_policy_path}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        wassert(output, output[1] == 0)
        print("\t --Generated input graph for reduction + policy")

    if not os.path.exists(reduction_user_path):
        command = f"python examples/synthetic/graphs/generate_reduce_graph.py -overlap 1 "
        command += f"-level 8 -branch 2 -N 6250 -gil_time 0 -weight 16000 "
        command += f"-user 1 -output {reduction_user_path}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        wassert(output, output[1] == 0)
        print("\t --Generated input graph for reduction + user")

    sub_dict = {}
    print("\t   [Running 1/3] Manual Movement, User Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {reduction_user_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 1 -user 1"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["m,u"] = sub_dict

    sub_dict = {}
    print("\t   [Running 2/3] Automatic Movement, User Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {reduction_user_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 2 -user 1"
        #print(command)
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,u"] = sub_dict

    sub_dict = {}
    print("\t   [Running 3/3] Automatic Movement, Policy Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {reduction_policy_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 2 -user 0"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #print(output)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict

    return output_dict


#Figure 9: Synthetic Independent
def run_independent(gpu_list, timeout):
    """
    Figure 9. Run Independent on the GPUs.
    """

    output_dict = {}
    sub_dict = {}

    # Generate input file
    if not os.path.exists("examples/synthetic/inputs"):
        print("\t  --Making input directory...")
        os.makedirs("examples/synthetic/inputs")
        print("\t  --Made input directory.")

    independent_policy_path = "examples/synthetic/inputs/independent_gpu_policy.txt"
    independent_user_path = "examples/synthetic/inputs/independent_gpu_user.txt"
    if not os.path.exists(independent_policy_path):
        command = f"python examples/synthetic/graphs/generate_independent_graph.py -overlap 0 "
        command += f"-width 300 -N 6250 -gil_time 0 -location 1 -weight 16000 "
        command += f"-user 0 -output {independent_policy_path}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        wassert(output, output[1] == 0)
        print("\t --Generated input graph for independent + policy")

    if not os.path.exists(independent_user_path):
        command = f"python examples/synthetic/graphs/generate_independent_graph.py -overlap 0 "
        command += f"-width 300 -N 6250 -gil_time 0 -location 1 -weight 16000 "
        command += f"-user 1 -output {independent_user_path}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        wassert(output, output[1] == 0)
        print("\t --Generated input graph for independent + user")

    sub_dict = {}
    print("\t   [Running 1/3] Manual Movement, User Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {independent_user_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 1 -user 1"
        print(command)
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["m,u"] = sub_dict

    sub_dict = {}
    print("\t   [Running 2/3] Automatic Movement, User Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {independent_user_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 2 -user 1"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,u"] = sub_dict

    sub_dict = {}
    print("\t   [Running 3/3] Automatic Movement, Policy Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {independent_policy_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 2 -user 0"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #print(output)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict

    return output_dict

#Figure 9: Synthetic Serial
def run_serial(gpu_list, timeout):
    """
    Figure 9. Run Serial on the GPUs.
    """

    output_dict = {}
    sub_dict = {}

    # Generate input file
    if not os.path.exists("examples/synthetic/inputs"):
        print("\t  --Making input directory...")
        os.makedirs("examples/synthetic/inputs")
        print("\t  --Made input directory.")

    serial_policy_path = "examples/synthetic/inputs/serial_gpu_policy.txt"
    serial_user_path = "examples/synthetic/inputs/serial_gpu_user.txt"
    if not os.path.exists(serial_policy_path):
        command = f"python examples/synthetic/graphs/generate_serial_graph.py -overlap 1 "
        command += f"-level 150 -N 6250 -gil_time 0 -location 1 -weight 16000 "
        command += f"-user 0 -output {serial_policy_path}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        wassert(output, output[1] == 0)
        print("\t --Generated input graph for serial + policy")

    if not os.path.exists(serial_user_path):
        command = f"python examples/synthetic/graphs/generate_serial_graph.py -overlap 1 "
        command += f"-level 150 -N 6250 -gil_time 0 -location 1 -weight 16000 "
        command += f"-user 1 -output {serial_user_path}"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        print(output)
        wassert(output, output[1] == 0)
        print("\t --Generated input graph for serial + user")

    sub_dict = {}
    print("\t   [Running 1/3] Manual Movement, User Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {serial_user_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 1 -user 1"
        print(command)
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        print(output)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["m,u"] = sub_dict

    sub_dict = {}
    print("\t   [Running 2/3] Automatic Movement, User Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {serial_user_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 2 -user 1"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,u"] = sub_dict

    sub_dict = {}
    print("\t   [Running 3/3] Automatic Movement, Policy Placement")
    for n_gpus in gpu_list:
        cuda_visible_devices = list(range(n_gpus))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        #print(f"Resetting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        command = f"python examples/synthetic/run.py -graph {serial_policy_path}"
        command += f" -d 1000 -loop 6 -reinit 2 -data_move 2 -user 0"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #print(output)
        times = parse_synthetic_times(output[0])
        print(f"\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    output_dict["a,p"] = sub_dict

    return output_dict

#Figure 14: Batched Cholesky Variants
def run_batched_cholesky(gpu_list, timeout):
    """
    Figure 14. Run Batched Cholesky on the GPUs. (Test variants)
    """
    cpu_dict = {}

    print("\t   [Running 1/2] CPU Support")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/variants/batched_cholesky.py -ngpus {n_gpus} -use_cpu 1"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    cpu_dict[1] = sub_dict

    print("\t   [Running 2/2] GPU Only")
    sub_dict = {}
    for n_gpus in gpu_list:
        command = f"python examples/variants/batched_cholesky.py -ngpus {n_gpus} -use_cpu 0"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_times(output[0])
        print(f"\t\t    {n_gpus} GPUs: {times}")
        sub_dict[n_gpus] = times
    cpu_dict[0] = sub_dict

    return cpu_dict


#Figure 11: Prefetching Plot
def run_prefetching_test(gpu_list, timeout):
    """
    Figure 11. Run Prefetching Test on the GPUs. (Test data movement)
    """
    auto_dict = {}
    data_sizes = [2, 4, 40]
    data_map = ["32MB", "64MB", "640MB"]
    sub_dict = {}
    idx = 0
    print("\t   [Running 1/2] Manual Movement")
    for data_size in data_sizes:
        command = f"python examples/synthetic/run.py -graph examples/synthetic/artifact/graphs/prefetch.gph -data_move 1 -loop 5 -d {data_size} -reinit 2"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_synthetic_times(output[0])
        print(f"\t {data_map[idx]} Data: {times}")
        sub_dict[data_size] = times
        idx += 1
    auto_dict["m"] = sub_dict

    sub_dict = {}
    idx = 0
    print("\t   [Running 2/2] Automatic Movement")
    for data_size in data_sizes:
        command = f"python examples/synthetic/run.py -graph examples/synthetic/artifact/graphs/prefetch.gph -data_move 2 -loop 5 -d {data_size} -reinit 2 -loop 5"
        output = pe.run(command, timeout=timeout, withexitstatus=True)
        #Make sure no errors or timeout were thrown
        wassert(output, output[1] == 0)
        #Parse output
        times = parse_synthetic_times(output[0])
        print(f"\t {data_map[idx]} Data: {times}")
        sub_dict[data_size] = times
        idx += 1
    auto_dict["a"] = sub_dict

    return auto_dict

#Figure 13: Independent Parla Scaling
def run_independent_parla_scaling(thread_list, timeout):
    """
    Figure 13. Run Independent Parla Scaling on the CPU. (Tests overhead)
    """

    n = 1000

    #NOTE: Task sizes in microseconds.
    #These are set manually. The full test takes a long time.

    #sizes = [800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
    sizes = [1600, 6400, 51200]

    size_dict = {}
    count = 1
    for size in sizes:
        thread_dict = {}
        print(f"\t   [Running {count}/{len(sizes)}], Task Grain = {size} μs")
        for thread in thread_list:
            command = f"python examples/synthetic/run.py -graph examples/synthetic/artifact/graphs/independent_1000.gph -threads {thread} -data_move 0 -weight {size} -use_gpu 0"
            output = pe.run(command, timeout=timeout, withexitstatus=True)
            #Make sure no errors or timeout were thrown
            #print(output)
            wassert(output, output[1] == 0)
            #Parse output
            times = parse_synthetic_times(output[0])
            print(f"\t   {size} μs, {thread} threads, {times}")
            thread_dict[thread] = times
        size_dict[size] = thread_dict
        count+=1
    return size_dict

#Figure 13: Independnet Dask Scaling
def run_independent_dask_thread_scaling(thread_list, timeout):
    """
    Figure 13. Run Independent Dask Thread Scaling on the CPU. (Tests overhead)
    """
    n = 1000

    #NOTE: Task sizes in microseconds.
    #These are set manually. The full test takes a long time.
    #sizes = [800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
    sizes = [1600, 6400, 51200]
    size_dict = {}
    count = 1
    for size in sizes:
        thread_dict = {}
        print(f"\t   [Running {count}/{len(sizes)}], Task Grain = {size} μs")
        for thread in thread_list:
            command = f"python examples/synthetic/artifact/scripts/run_dask_thread.py -workers {thread} -time {size} -n {n}"
            output = pe.run(command, timeout=timeout, withexitstatus=True)
            #Make sure no errors or timeout were thrown
            wassert(output, output[1] == 0)
            #Parse output
            times = parse_times(output[0])

            print(f"\t   {size} μs, {thread} threads, {times}")
            thread_dict[thread] = times
        size_dict[size] = thread_dict
    return size_dict

#Figure 13: Independent Dask Scaling
def run_independent_dask_process_scaling(process_list, timeout):
    """
    Figure 13. Run Independent Dask Process Scaling on the CPU. (Tests overhead)
    """
    n = 1000

    #NOTE: Task sizes in microseconds.
    #These are set manually. The full test takes a long time.
    #sizes = [800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
    sizes = [1600, 6400, 51200]
    process_list = [1, 4, 8, 16, 32, 36, 40]
    size_dict = {}
    count = 1
    for size in sizes:
        process_dict = {}
        inner_count = 0
        print(f"\t   [Running {count}/{len(sizes)}], Task Grain = {size} μs")
        for process in process_list:
            command = "rm -rf dask-worker-space"
            output = pe.run(command, timeout=None)
            command = f"python examples/synthetic/artifact/scripts/run_dask_process.py -workers {process} -time {size} -n {n}"
            print("LIMIT: ", timeout)
            output = pe.run(command, timeout=timeout, withexitstatus=True)
            #Make sure no errors or timeout were thrown
            #wassert(output, output[1] == 0, verbose=True, require=False)
            print("OUTPUT: ", output)
            #Parse output
            if output[1] == None:
                times = "TIMEOUT"
                process_dict[process] = times
                print(f"\t   {size} μs, {process} processes, {times}")
                break
            else:
                times = parse_times(output[0])
                print(f"\t   {size} μs, {process} processes, {times}")
                if inner_count == 0:
                    timeout = 5*times[0]

                process_dict[process] = times

            inner_count += 1
        size_dict[size] = process_dict
    return size_dict

#Figure 13: GIL test (Parla)
def run_GIL_test_parla(thread_list, timeout):
    """
    Figure 13. Run Independent Parla Scaling on the CPU. (Tests overhead).
    This uses the interface to set time spent holding the GIL.
    """
    n = 1000
    sizes = [45000]
    gil = 5000
    size_dict = {}
    count = 1
    for size in sizes:
        thread_dict = {}
        print(f"\t   [Running {count}/{len(sizes)}], Task Grain = {size} μs")
        for thread in thread_list:
            command = f"python examples/synthetic/run.py -graph examples/synthetic/artifact/graphs/independent_1000.gph -threads {thread} -data_move 0 -weight {size} -use_gpu 0 -gweight {gil}"
            output = pe.run(command, timeout=timeout, withexitstatus=True)
            #Make sure no errors or timeout were thrown
            print(output)
            wassert(output, output[1] == 0)
            #Parse output
            times = parse_synthetic_times(output[0])
            print(f"\t   {size} μs, {thread} threads, {times}")
            thread_dict[thread] = times
        size_dict[size] = thread_dict
        count+=1
    return size_dict

#Figure 13: GIL test (Dask Threading)
def run_GIL_test_dask(thread_list, timeout):
    """
    Figure 13. Run Independent Dask Thread Scaling on the CPU. (Tests overhead).
    This uses the interface to set time spent holding the GIL.
    """
    n = 1000
    sizes = [50000]
    gil = 5000
    size_dict = {}
    count = 1
    for size in sizes:
        thread_dict = {}
        print(f"\t   [Running {count}/{len(sizes)}], Task Grain = {size} μs")
        for thread in thread_list:
            command = f"python examples/synthetic/artifact/scripts/run_dask_thread_gil.py -workers {thread} -time {size} -n {n}"
            output = pe.run(command, timeout=timeout, withexitstatus=True)
            #Make sure no errors or timeout were thrown
            wassert(output, output[1] == 0)
            #Parse output
            times = parse_times(output[0])

            print(f"\t   {size} μs, {thread} threads, {times}")
            thread_dict[thread] = times
        size_dict[size] = thread_dict
    return size_dict
#NOTE: Change this to test a single experiment in isolation. It is the default execution list.
test = [run_serial]

figure_9 = [run_jacobi, run_matmul, run_blr_parla, run_blr_threads, run_nbody_parla, run_nbody_threads, run_reduction, run_independent, run_serial]
figure_9_cublas = [run_matmul_cublas]
figure_9_magma = [run_cholesky_magma]

figure_11 = [run_prefetching_test]

figure_12 = [run_cholesky_20_host, run_cholesky_20_gpu]
figure_12_dask = [run_dask_cholesky_20_host, run_dask_cholesky_20_gpu]

figure_13 = [run_independent_parla_scaling, run_GIL_test_parla]
figure_13_dask = [run_independent_dask_thread_scaling, run_independent_dask_process_scaling, run_GIL_test_dask]

figure_14 = [run_batched_cholesky]

figure_dictionary = {}
figure_dictionary['Figure_9'] = figure_9
figure_dictionary['Figure_9_cublas'] = figure_9_cublas
figure_dictionary['Figure_9_magma'] = figure_9_magma

figure_dictionary['Figure_12'] = figure_12
figure_dictionary['Figure_12_dask'] = figure_12_dask

figure_dictionary['Figure_14'] = figure_14

figure_dictionary['Figure_11'] = figure_11

figure_dictionary['Figure_13'] = figure_13
figure_dictionary['Figure_13_dask'] = figure_13_dask

figure_dictionary['Figure_test'] = test

figure_output = {}

if __name__ == '__main__':
    import os
    import sys
    parser = argparse.ArgumentParser(description='Runs the benchmarks')
    parser.add_argument('--figures', type=str, nargs="+",
        help='Figure numbers to test (9, 9_cublas, 9_magma, 11, 12, 12_dask, 13, 13_dask, 14). \
              Execution time expectations: 9 (1 min), 9_magma (2 min), 11 (1 min), 12 (21 min), \
              12_dask (23 min), 13 (7 min), 13_dask (9 min), 14 (8 min)', default=None)
    parser.add_argument('--timeout', type=int, help='Max Timeout for a benchmark', default=1000)

    args = parser.parse_args()

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is None:
        print("Warning CUDA_VISIBLE_DEVICES is not set.")
        cuda_visible_devices = list(range(4))
        cuda_visible_devices = ','.join(map(str, cuda_visible_devices))
        print(f"Setting CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)

    print("If running synthetic or 3rd party benchmarks, make sure all \
          executables and submodules are compiled. See artifact/README and \
          examples/synthetic/README for more information.")

    #NOTE: Set ngpus and nthreads for your machine here
    ngpus = [1, 2, 4]
    nthreads = [1, 2, 4, 8, 16]

    if args.figures is None:
        figure_list = ["-1"]
    else:
        figure_list = args.figures

    for figure_num in figure_list:
        device_list = ngpus
        if figure_num.lstrip('-').isdigit():
            if int(figure_num) < 0:
                figure_num = "test"
            elif int(figure_num) == 13:
                device_list = nthreads

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
            output_dict = test(device_list, args.timeout)
            test_output[test.__name__] = output_dict
            #print("\t --Experiment {}/{} Completed. Output: {}".format(i, total_tests, output_dict))
            print("\t --Experiment {}/{} Completed.".format(i, total_tests))
            i += 1
        figure_output[figure] = test_output
        print(f"Collection for {figure} complete")


    print("All experiments complete.")
    print("Output:")
    print(figure_output)
