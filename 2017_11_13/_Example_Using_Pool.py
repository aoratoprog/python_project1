import time
import logging

import multiprocessing
from multiprocessing import Process, Queue, Pool

multiprocessing.log_to_stderr()
logger = multiprocessing.get_logger()
logger.setLevel(logging.INFO)

startTime = time.time()

def fun1(a):
    a = a*a
    a = a*a
    a = a*a
    a = a*a
    # result.put(a)
    return a

def fun2(a):
    # result.put(a)
    return a

num = 2
pool = Pool(processes=2)

r1 = pool.apply_async(fun1, args=(num,))

var1 = r1.get()

pool.close()
pool.join()

print(var1)

endTime = time.time() - startTime
print(endTime)