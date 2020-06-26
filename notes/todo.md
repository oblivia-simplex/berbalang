# TODO

- set up graphing/plotting
- set up neo4j

- optimize!
- make it easier to switch between weighted and pareto (and lexicographic) fitness vectors.
ideally, this is somethign you should be able to set in the config.toml

- make sure linear_gp works. probably a few kinks to work out. 

- implement a ROPush machine (finish -- genetic operators for ropush)

- new fitness factor: bytes written. reinstate mem write hook.
- better: reward if one of the indirect values in the register pattern is written.
- reward bytes written to the extent that they approximate an indirect rp value

- consider treating the register values lexicographically. 
  use full lexicographic selection, perhaps. reread the lexicographic paper. 
  supposedly works well for problems that require exact solutions. 
  
  this might warrant implementing a new impl of the Epoch, alongside Tournament
  and Roulette, etc. iirc, the selection procedure is quite different. There's
  a rather shoddy attempt at doing this in GENLIN, though I didn't have much
  luck over there.  
  
- TTL on genes. See appendix to thesis. how did this work?

- it would be interesting to try rewarding chains that read memory that they themselves have written. bit complex. might approximate by ignoring sequence. might not. all the info is available in the hatchery after all. 

    track addresses read from, and then later check intersection with addresses written to. abstract away from sequence. it's about building up useful material and control. 

- PE files need support

- profile the hatchery. how many workers and engines are sitting idle? how much time is spent blocking?

- new task: write a given string somewhere in memory
- 2nd stage: point to it with a register

- dockerize
-- started this. having some issues with the network. trouble accessing github in the rust build. troubleshoot this. 

- data scraping and processing tools for results. plot graphs automatically. populate visdom, maybe.


- get it cloud ready

- Add support for *staged* objectives, or sub-tasks. There's a fair bit of GP lit on this, I think. Take a look. Look at what Malcom did with the soccer task, e.g.

- for exacting tasks, run n evaluations with different (constant) randomized states

## System call constraints
structural features as fitness attributes
see the thread in the slack with peli about this

- improve the loader by forking some code from falcon
the falcon elf loader is very nice and easily usable 
 
 