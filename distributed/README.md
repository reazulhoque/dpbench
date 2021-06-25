# Benchmarks for distributed numpy implementations

Each benchmark is implemented using different distributed implementations of numpy. For baseline comparison versions using torch and plain numpy are also provided.
The following distributed implementataions are available
* NumS
* legate
* HeAT
* dask-array

The top level directory holds running scripts `<benchname>_run.py` which allow selecting the implementation through the argument `-u`or `--use` followed by one of the above tools.
The runner scripts use the implementations in the respective benchmark directories.
The runner scripts do not setup clusters (ray, dask etc) or use a launcher like mpirun or legate. This is currently in the responsibility of the user.

We currently have the following benchmarks
* jstencil: Jacobi stencil using shifted whole-array operations
* lbfgs: Limited-memory Broyden–Fletcher–Goldfarb–Shannon
         Use lbfgs/gen_Xy_data.py to generate input data
* linreg: Linear Regression
