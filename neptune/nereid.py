#hacky version of an experiment pusher for neptune.ai,needs a bunch of local files, but this should be obvious
#Nereid is one of the two larger moons of Neptune, the biggest is Triton. 
#...but thats and entirely other script


import neptune
import toml
import os,sys
import numpy as np
import simplejson as json

def read_toml(path):
#Read in the config.toml for a specific berbalang tournament

   cf = toml.load(path)
   return cf

#Read in the .json log of a champion from a specific berbalang island
def read_berb_log(path):
  
   with open(path) as log:
      jl=json.load(log)
   
   return jl

def build_experiment(conf,logpath,resultname,csvfile):

   with open(csvfile) as stats_f:
       params=stats_f.readline().strip().split(",")
       stats=stats_f.readlines()
   
   for s in stats:
      elems=s.strip().split(",")
      counter,scalar_fit,priority_fit=[elems[0],elems[4],elems[5]]
      print(counter,scalar_fit,priority_fit)

   exp_config = read_toml(conf)
   exp_number=1

   champ = read_berb_log(logpath)
   exp_name=champ['chromosome']['name']
   exp_desc=str(champ['tag'])

   exp_params={"some_param":0.1,"other_param":128,"yet_another_param":31337 }
 
   exp_log_artifact = ["data/champion_statistics.csv","mean_statistics.csv" ]

#Neptune init
   neptune.init('special-circumstances/sandbox',api_token=None)

   neptune.create_experiment(name=exp_name,params=exp_params)

   for s in stats:
     elems=s.strip().split(",")
     counter,scalar_fit,priority_fit=[elems[0],elems[4],elems[5]]
     neptune.log_metric(params[0],int(counter))
     neptune.log_metric(params[4],float(scalar_fit))
     neptune.log_metric(params[5],float(priority_fit))
    
   neptune.log_image('pleasures_1',"/home/armadilo/projects/neptune/data/clamp-liked-zeros-count-pleasures.png")
   neptune.log_image('pleasures_2',"/home/armadilo/projects/neptune/data/lamas-koala-zero-count-pleasures.png")
   neptune.send_artifact('/home/armadilo/projects/neptune/data/champion_statistics.csv') 
   neptune.send_artifact('/home/armadilo/projects/neptune/data/mean_statistics.csv')
 
if __name__ == "__main__":

   build_experiment("/home/armadilo/logs/berbalang/Roper/Tournament/2020/08/26/config-0/config.toml","/home/armadilo/logs/berbalang/Roper/Tournament/2020/08/26/config-0/island_0/champions/champion_12288.json","./experiment.json","./data/mean_statistics.csv")  
