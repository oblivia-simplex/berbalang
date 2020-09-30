import neptune
import toml
import os,sys
import numpy as np
import simplejson as json
import time
import tailer
import subprocess
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class Triton:

   def __init__(self,watch_path,island):

       self.session_config_path = ""
       self.config_toml=""
       self.log_path = watch_path
       self.island_num=island
       self.exp_params={}
       self.exp_project_path="special-circumstances/sandbox"

   def read_toml(path):

      self.config_toml = toml.load(path)

   def read_berb_log(path):

      with open(path) as log:
         jl=json.load(log)
      return jl

   #def build_experiment(conf,logpath,resultname):
    def build_experiment(conf,resultname):
      exp_config = self.config_toml
      exp_number=1
      #This is a bit messy, but more literal
      exp_params={'num_islands':exp_config['num_islands'],'mutation_rate':exp_config['mutation_rate'],
                  'mutation_exponent':exp_config['mutation_exponent'],
                  'crossover_period':exp_config['crossover_period'],
                  'crossover_rate':exp_config['crossover_rate'],
                  'max_init_len':exp_config['max_init_len'],
                  'min_init_len':exp_config['min_init_len'],
                  'pop_size':exp_config['pop_size'],
                  'max_length':exp_config['max_length']
                  }
      }

      #champ = read_berb_log(self.log_path)
      #exp_name=champ['chromosome']['name']
      #exp_desc=str(champ['tag'])
      exp_name=exp_config['population_name']
      exp_properties={"data_version":exp_config['population_name'],"data_path":"/home/armadilo/projects/berbalang/agaricus-lepiota.tsv"}
      exp_tags=["sometag","some_other_tag"]
      exp_source="./trials.sh"
      }
      with open(resultname,'w') as log_f:
          json.dump(experiment,log_f)

   def on_created(event):
       print(f"{event.src_path}" has been created")
       if event.is_directory:
       #print(f"{event.src_path} has been created")
          #if "island" in event.src_path:
            # print(f"{event.src_path} has been created")

        #config-0 at the top-level of a log tree indicates an experiment
        #has started to run. Call build_experiment to create it in neptune
          if "config-0" in event.src_path:
             self.session_config_path = event.src_path+"/config.toml"
             self.read_toml(self.session_config_path)
             self.build_experiment()
             build_experiment("/home/armadilo/projects/berbalang/config.toml","home/armadilo/projects/berbalang/neptune/experiment.json")

          #If a csv file is created in the island dir
          #Spin up a tailf subprocess
          if event.src_path.endswith(".csv"):
              self.tailer(event.src_path,self.)


       else:
          if ".json" in event.src_path:
             print(f"Found pre-compressed json file: {event.src_path} ")

   def on_modified(event):
       print(f"{event.src_path} has been modified")

   if __name__ == "__main__":

       build_experiment("/home/armadilo/projects/berbalang/config.toml","/home/armadilo/logs/berbalang/Roper/Tournament/2020/08/26/config-0/island_0/champions/champion_12288.json","./experiment.json")
