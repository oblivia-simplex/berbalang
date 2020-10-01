import neptune,triton
import toml
import os,sys
import numpy as np
import simplejson as json
import time
import tailer
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

write_count=0
current_epoch=0
island="island_0"

def on_created(event):
   if event.is_directory:
   #config-0 at the top-level of a log tree indicates an experiment
   #has started to run. Call build_experiment to create it in neptune
      if event.src_path.endswith("config-0"):
         print(f"{event.src_path} has been created")
         session_config_path = event.src_path+"/config.toml"
         triton.read_toml(session_config_path)
         triton.build_experiment()

def on_modified(event):
   if event.src_path.endswith(".csv") and island in event.src_path:
      print(f"{event.src_path} has been modified")
      logpath=event.src_path
      #for w in watchlist:
      #   if w in logpath:
      if "mean" in logpath:
            logline=tailer.tail(open(logpath),1)[0]
            stats=[float(l) for l in logline.strip().split(",")]
            triton.log_stats_to_experiment(stats,logpath)

if __name__ == "__main__":
    # create the event handler
    patterns="*.csv"
    ignore_patterns = ["^./.git"]
    ignore_directories = False
    case_sensitive = True
#    my_event_handler = RegexMatchingEventHandler(ignore_regexes=ignore_patterns, ignore_directories=ignore_directories, case_sensitive=case_sensitive)
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

    my_event_handler.on_created = on_created
    my_event_handler.on_modified = on_modified

    # create an observer
    path = "/home/armadilo/logs/berbalang/Roper/Tournament"
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    triton = triton.Triton(path,0,"special-circumstances","sandbox")

    my_observer.start()
    try:
        while True:
            time.sleep(5)
    except:
        my_observer.stop()
        print("Observer Stopped")
    my_observer.join()
