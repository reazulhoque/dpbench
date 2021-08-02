import numpy as np
import numba_dppy
import dpctl
import dpctl.tensor as dpt

try:
    import itimer as it
    now = it.itime
except:
    from timeit import default_timer
    now = default_timer


@numba_dppy.kernel
def atomic_add(a):
    idx = numba_dppy.get_global_id(0)
    numba_dppy.atomic.add(a, idx, 1)


global_size = 1024 * 1024 * 500
alignment = 64

d = dpctl.SyclDevice("opencl:gpu")
sub_devs = d.create_sub_devices(partition="next_partitionable")
device_ctx = dpctl.SyclContext(sub_devs[0])
queue = dpctl.SyclQueue(device_ctx, sub_devs[0])

dtyp = np.float32
hb = np.zeros(global_size, dtype=dtyp)
itemsize = np.dtype(dtyp).itemsize

'''
usm_data = dpctl.memory.MemoryUSMShared(global_size * itemsize, alignment=alignment, queue=queue)
usm_data.copy_from_host(hb.view("B"))
data = np.ndarray(hb.shape, buffer=usm_data, dtype=hb.dtype)
np.copyto(data, hb)


'''
data = dpt.usm_ndarray(hb.shape, dtype=hb.dtype, buffer="device", buffer_ctor_kwargs={"alignment": alignment, "queue": queue})
data.usm_data.copy_from_host(hb.reshape((-1)).view("|u1"))


print("N =", global_size)
print("ALIGNMENT =", alignment)

with dpctl.device_context(queue):
    for i in range(3):
        t0 = now()
        atomic_add[global_size, numba_dppy.DEFAULT_LOCAL_SIZE](data)
        time = now() - t0

        print(("Total Time = %s seconds") % time)
