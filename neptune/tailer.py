import tailf
import time
import sys
import neptune

params=["counter","epoch","length","register_error","scalar_fitness","priority_fitness","uniq_exec_count","ratio_written","emulation_time","ratio_eligible"]

class Tailer:

def __init__(pathname,project,experiment):

   self.project = project
   self.experiment = experiment

def tailer(pathname):

   with tailf.Tail(pathname) as tail:
       while True:
           for event in tail:
               if isinstance(event, bytes):
                   pusher(str(event.decode("utf-8"))
                   print(event.decode("utf-8"), end='')
               elif event is tailf.Truncated:
                   print("File was truncated")
               else:
                   assert False, "unreachable" # currently. more events may be introduced later
           time.sleep(0.01) # save CPU cycles

def pusher(logline):

   stats = logline.strip().split(",")
   for p in range(len(params)):
      self.experiment.log_metric(params[p],stats[p])

#if __name__ == "__main__":
#
#  tailpath=sys.argv[1] 
#  tailer(tailpath)
