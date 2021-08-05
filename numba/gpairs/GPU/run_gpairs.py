import base_gpairs
import numpy as np
import gaussian_weighted_pair_counts as gwpc
import numba_dppy
import dpctl
import dpctl.tensor as dpt
# from gpairs.pair_counter.tests.generate_test_data import (
#     DEFAULT_RBINS_SQUARED)

# DEFAULT_NBINS = 20
# DEFAULT_RMIN, DEFAULT_RMAX = 0.1, 50
# DEFAULT_RBINS = np.logspace(
#     np.log10(DEFAULT_RMIN), np.log10(DEFAULT_RMAX), DEFAULT_NBINS).astype(
#         np.float32)
# DEFAULT_RBINS_SQUARED = (DEFAULT_RBINS**2).astype(np.float32)

def run_gpairs(x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared):
    blocks = 512

    result = np.zeros_like(rbins_squared)[:-1]
    result = result.astype(np.float64)

    with dpctl.device_context(base_gpairs.get_device_selector()) as gpu_queue:
        # gwpc.count_weighted_pairs_3d_intel[blocks, numba_dppy.DEFAULT_LOCAL_SIZE](
        #     d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
        #     d_rbins_squared, result)

        d_x1 = dpt.usm_ndarray(x1.shape, dtype=x1.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_x1.usm_data.copy_from_host(x1.reshape((-1)).view("|u1"))

        d_y1 = dpt.usm_ndarray(y1.shape, dtype=y1.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_y1.usm_data.copy_from_host(y1.reshape((-1)).view("|u1"))

        d_z1 = dpt.usm_ndarray(z1.shape, dtype=z1.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_z1.usm_data.copy_from_host(z1.reshape((-1)).view("|u1"))

        d_w1 = dpt.usm_ndarray(w1.shape, dtype=w1.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_w1.usm_data.copy_from_host(w1.reshape((-1)).view("|u1"))

        d_x2 = dpt.usm_ndarray(x2.shape, dtype=x2.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_x2.usm_data.copy_from_host(x2.reshape((-1)).view("|u1"))

        d_y2 = dpt.usm_ndarray(y2.shape, dtype=y2.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_y2.usm_data.copy_from_host(y2.reshape((-1)).view("|u1"))

        d_z2 = dpt.usm_ndarray(z2.shape, dtype=z2.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_z2.usm_data.copy_from_host(z2.reshape((-1)).view("|u1"))

        d_w2 = dpt.usm_ndarray(w2.shape, dtype=w2.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_w2.usm_data.copy_from_host(w2.reshape((-1)).view("|u1"))

        d_rbins_squared = dpt.usm_ndarray(rbins_squared.shape, dtype=rbins_squared.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_rbins_squared.usm_data.copy_from_host(rbins_squared.reshape((-1)).view("|u1"))

        d_result = dpt.usm_ndarray(result.shape, dtype=result.dtype, buffer="device", buffer_ctor_kwargs={"queue": gpu_queue})
        d_result.usm_data.copy_from_host(result.reshape((-1)).view("|u1"))


        gwpc.count_weighted_pairs_3d_intel_ver2[x1.shape[0], numba_dppy.DEFAULT_LOCAL_SIZE](
            d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2,
            d_rbins_squared, d_result)

        d_result.usm_data.copy_to_host(result.reshape((-1)).view("|u1"))
        
base_gpairs.run("Gpairs Dppy kernel",run_gpairs)
