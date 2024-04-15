import pynvml
import os
import time
from functools import reduce
import datetime


pynvml.nvmlInit()


def allow_execution(gpu_id, memory=0.85):
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu_id))
    if isinstance(memory,float) and memory<1.0:
        mem_require = meminfo.total * memory / 1e9
    else:
        mem_require = memory
    return mem_require, meminfo.free / 1e9


def wait_for_allow_execution(gpu_cnt=1,memory=0.85):
    '''gpu_cnt: count of gpus needed for program
       memory: 0~1 for percentage and >1 for memory'''
    start = time.time()
    print("###auto_start: Wait for start...")
    gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].strip().split(",")]

    avail_gpus=0
    avail_gpu_ids=[]
    for gpu in gpu_ids:
        mem_require, mem_free = allow_execution(gpu, memory)
        if mem_require < mem_free:
            print(f"gpu{gpu}: need {mem_require:.2f}GB<free {mem_free:.2f}GB")
            avail_gpus += 1
            avail_gpu_ids.append(gpu)
        else:
            print(f"gpu{gpu}: need {mem_require:.2f}GB>free {mem_free:.2f}GB")
    print('%d gpus available. Needed %d.'%(avail_gpus,gpu_cnt))

    while not avail_gpus>=gpu_cnt:
        avail_gpus = 0
        avail_gpu_ids=[]
        for gpu in gpu_ids:
            mem_require, mem_free = allow_execution(gpu, memory)
            if mem_require < mem_free:
                avail_gpus += 1
                avail_gpu_ids.append(gpu)
                if avail_gpus>=gpu_cnt:
                    break
        time.sleep(60)

    print("###auto_start: Start on gpu: %s"%",".join(map(str,avail_gpu_ids)))
    print("## Current time is: ",str(datetime.datetime.now()),
          "; Waited for %.2f hours"%((time.time()-start)/3600))
    return ",".join(map(str,avail_gpu_ids))
    


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2,3,5,6,7'
    wait_for_allow_execution(gpu_cnt=3,memory=0.1)