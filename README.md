# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py



# Parallel Check

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (163)
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        in_storage: Storage,                                             |
        in_shape: Shape,                                                 |
        in_strides: Strides,                                             |
    ) -> None:                                                           |
                                                                         |
        for i in prange(len(out)):---------------------------------------| #0
            out_index = np.empty(len(in_shape), np.int32)                |
            in_index = np.empty(len(in_shape), np.int32)                 |
            to_index(i, out_shape, out_index)                            |
            broadcast_index(out_index, out_shape, in_shape, in_index)    |
            in_pos = index_to_position(in_index, in_strides)             |
            out[i] = fn(in_storage[in_pos])                              |
        # TODO: Implement for Task 3.1.                                  |
        # raise NotImplementedError("Need to implement for Task 3.1")    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (173) is
hoisted out of the parallel loop labelled #0 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (174) is
hoisted out of the parallel loop labelled #0 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (208)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (208)
-------------------------------------------------------------------------|loop #ID
    def _zip(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        a_storage: Storage,                                              |
        a_shape: Shape,                                                  |
        a_strides: Strides,                                              |
        b_storage: Storage,                                              |
        b_shape: Shape,                                                  |
        b_strides: Strides,                                              |
    ) -> None:                                                           |
        for i in prange(len(out)):---------------------------------------| #1
            out_index = np.empty(len(out_shape), np.int32)               |
            a_index = np.empty(len(a_shape), np.int32)                   |
            b_index = np.empty(len(b_shape), np.int32)                   |
            to_index(i, out_shape, out_index)                            |
            broadcast_index(out_index, out_shape, a_shape, a_index)      |
            broadcast_index(out_index, out_shape, b_shape, b_index)      |
            a_idx = index_to_position(a_index, a_strides)                |
            b_idx = index_to_position(b_index, b_strides)                |
            out[i] = fn(a_storage[a_idx], b_storage[b_idx])              |
                                                                         |
                                                                         |
        # TODO: Implement for Task 3.1.                                  |
        # raise NotImplementedError("Need to implement for Task 3.1")    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (220) is
hoisted out of the parallel loop labelled #1 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (221) is
hoisted out of the parallel loop labelled #1 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (222) is
hoisted out of the parallel loop labelled #1 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (258)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (258)
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        a_storage: Storage,                                              |
        a_shape: Shape,                                                  |
        a_strides: Strides,                                              |
        reduce_dim: int,                                                 |
    ) -> None:                                                           |
        reduce_size = a_shape[reduce_dim]                                |
        for i in prange(len(out)):---------------------------------------| #2
            out_index = np.empty(len(out_shape), np.int32)               |
            to_index(i, out_shape, out_index)                            |
            o = index_to_position(out_index, out_strides)                |
            for s in range(reduce_size):                                 |
                out_index[reduce_dim] = s                                |
                j = index_to_position(out_index, a_strides)              |
                out[o] = fn(out[o], a_storage[j])                        |
        # TODO: Implement for Task 3.1.                                  |
        # raise NotImplementedError("Need to implement for Task 3.1")    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (269) is
hoisted out of the parallel loop labelled #2 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (282)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\madhu\Documents\Cornell\mod3-Madhu2244\minitorch\fast_ops.py (282)
----------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                              |
    out: Storage,                                                                                         |
    out_shape: Shape,                                                                                     |
    out_strides: Strides,                                                                                 |
    a_storage: Storage,                                                                                   |
    a_shape: Shape,                                                                                       |
    a_strides: Strides,                                                                                   |
    b_storage: Storage,                                                                                   |
    b_shape: Shape,                                                                                       |
    b_strides: Strides,                                                                                   |
) -> None:                                                                                                |
    """NUMBA tensor matrix multiply function.                                                             |
                                                                                                          |
    Should work for any tensor shapes that broadcast as long as                                           |
                                                                                                          |
    ```                                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                                     |
    ```                                                                                                   |
                                                                                                          |
    Optimizations:                                                                                        |
                                                                                                          |
    * Outer loop in parallel                                                                              |
    * No index buffers or function calls                                                                  |
    * Inner loop should have no global writes, 1 multiply.                                                |
                                                                                                          |
                                                                                                          |
    Args:                                                                                                 |
    ----                                                                                                  |
        out (Storage): storage for `out` tensor                                                           |
        out_shape (Shape): shape for `out` tensor                                                         |
        out_strides (Strides): strides for `out` tensor                                                   |
        a_storage (Storage): storage for `a` tensor                                                       |
        a_shape (Shape): shape for `a` tensor                                                             |
        a_strides (Strides): strides for `a` tensor                                                       |
        b_storage (Storage): storage for `b` tensor                                                       |
        b_shape (Shape): shape for `b` tensor                                                             |
        b_strides (Strides): strides for `b` tensor                                                       |
                                                                                                          |
    Returns:                                                                                              |
    -------                                                                                               |
        None : Fills in `out`                                                                             |
                                                                                                          |
    """                                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                |
                                                                                                          |
    row_increment = a_strides[2]                                                                          |
    col_increment = b_strides[1]                                                                          |
    common_dim = a_shape[-1]                                                                              |
                                                                                                          |
    for row_i in prange(0, out_shape[0]):-----------------------------------------------------------------| #3
        for col_j in range(0, out_shape[1]):                                                              |
            for block_k in range(0, out_shape[2]):                                                        |
                row_start_index = row_i * a_batch_stride + col_j * a_strides[1]                           |
                col_start_index = row_i * b_batch_stride + block_k * b_strides[2]                         |
                                                                                                          |
                block_sum = 0.0                                                                           |
                                                                                                          |
                for block in range(0, common_dim):                                                        |
                    block_sum += a_storage[row_start_index] * b_storage[col_start_index]                  |
                    row_start_index += row_increment                                                      |
                    col_start_index += col_increment                                                      |
                                                                                                          |
                out_index = row_i * out_strides[0] + col_j * out_strides[1] + block_k * out_strides[2]    |
                out[out_index] = block_sum                                                                |
                                                                                                          |
    # TODO: Implement for Task 3.2.                                                                       |
    # raise NotImplementedError("Need to implement for Task 3.2")                                         |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
