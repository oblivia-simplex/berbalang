import neptune
import toml
import os,sys
import numpy as np
import simplejson as json
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

write_count=0
current_epoch=0

def on_created(event):
    if event.is_directory:
       #print(f"{event.src_path} has been created")
       if "island" in event.src_path:
          print(f"{event.src_path} has been created")
    else:
       if ".json" in event.src_path:
          print(f"Found pre-compressed json file: {event.src_path} ")

def on_deleted(event):
    print(f"Delete {event.src_path}!")


def on_modified(event):
    #print(f"{event.src_path} has been modified")
    if event.src_path.endswith(".gz") :
       print(f"({event.src_path} was modified")
       
if __name__ == "__main__":
    # create the event handler
    patterns="*.csv"
    ignore_patterns = ["^./.git"]
    ignore_directories = False
    case_sensitive = True
#    my_event_handler = RegexMatchingEventHandler(ignore_regexes=ignore_patterns, ignore_directories=ignore_directories, case_sensitive=case_sensitive)
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

    my_event_handler.on_created = on_created
    my_event_handler.on_deleted = on_deleted
    my_event_handler.on_modified = on_modified

    # create an observer
    path = "/home/armadilo/logs/berbalang/Roper/Tournament"
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    my_observer.start()
    try:
        while True:
            time.sleep(5)
    except:
        my_observer.stop()
        print("Observer Stopped")
    my_observer.join()
