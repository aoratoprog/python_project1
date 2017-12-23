import time
import logging

import multiprocessing
from multiprocessing import Process, Queue, Pool

multiprocessing.log_to_stderr()
logger = multiprocessing.get_logger()
logger.setLevel(logging.INFO)

startTime = time.time()

def fun1(a, result):
    a = a*a
    a = a*a
    a = a*a
    a = a*a
    result.put(a)

def fun2(a, result):
    result.put(a)

num = 2
result = Queue()

pr1 = Process(target=fun1, args=(num, result))
pr2 = Process(target=fun2, args=(num, result))

pr1.start()
pr2.start()

var1 = result.get()
var2 = result.get()

result.close()
pr1.join()
pr2.join()

print(var1, var2)

endTime = time.time() - startTime
print(endTime)