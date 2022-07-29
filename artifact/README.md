#Repositories for 3rd Party Performance Comparisons

## Cublasmg

For SC22 Artifact review, we provide Cublasmg here, which is also available from: https://developer.nvidia.com/cudamathlibraryea
Set `CUBLASMG_ROOT=PARLA_ROOT/artifact/cublasmg` and `CUDAMG_ROOT=PARLA_ROOT/artifact/cudalibmg`

`In `CUBLASMG_ROOT/cublasmg/test`, we have the modified block matrix multiplication file to perform: C = A @ B.T at the same size as Parla.

You must compile the examples in the test folder. (Requires CUDA and the set paths above)

```
cd CUBLASMG_ROOT/cublasmg/test
make
```

## Magma

For convience we include the MAGMA linear algebra library as a submodule.
Unless you have a local installation, set MAGMA_ROOT=PARLA_ROOT/artifact/magma

To run the cholesky comparison you must build magma with testing enabled.
To compare we use the testing/testing_dpotrf_mgpu executable with `-N 28000`
and the '--ngpu' flag.
Instructions for this are included in the MAGMA readme.
