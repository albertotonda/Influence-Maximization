import logging
import sys
IS_PY2 = sys.version_info < (3, 0)
 
if IS_PY2:
    from Queue import Queue
else:
    from queue import Queue
 
from threading import Thread
 
 
class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """
    def __init__(self, tasks, thread_id):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.id = thread_id
        self.start()
 
    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            logging.debug("[Thread %d] Args retrieved: \"%s\"" % (self.id, args))
            new_args = []
            logging.debug("[Thread %d] Length of args: %d" % (self.id, len(args)))
            for a in args[0] : new_args.append(a)
            new_args.append(self.id)
            logging.debug("[Thread %d] Length of new_args: %d" % (self.id, len(new_args)))
            try:
                func(*new_args, **kargs)
            except Exception as e:
                # An exception happened in this thread
                print(e)
            finally:
                # Mark this task as done, whether an exception happened or not
                logging.debug("[Thread %d] Task completed." % self.id)
                self.tasks.task_done()
 
 
class ThreadPool:
    """ Pool of threads consuming tasks from a queue """
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for i in range(num_threads):
            Worker(self.tasks, i)
 
    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))
 
    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)
 
    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()
 
 
# this main here is just to test the threadpool
if __name__ == "__main__":
    from random import randrange
    from time import sleep

    # initialize logging
    import logging
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S') 
 
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
 
    # Function to be executed in a thread
    def wait_delay(d, ninja, index, thread_id):
        logging.info("[Thread %d] is taking care of \"%s\" %d, it will take (%d)sec" % (thread_id, ninja, index, d))
        sleep(d)
 
    # Generate random delays
    delays = [ (randrange(3, 7), "ninja", i) for i in range(50)]
 
    # Instantiate a thread pool with 5 worker threads
    pool = ThreadPool(5)
 
    # Add the jobs in bulk to the thread pool. Alternatively you could use
    # `pool.add_task` to add single jobs. The code will block here, which
    # makes it possible to cancel the thread pool with an exception when
    # the currently running batch of workers is finished.
    pool.map(wait_delay, delays)
    pool.wait_completion()
