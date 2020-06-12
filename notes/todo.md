# TODO

- Basic block bigram diversity measure using a count min sketch.

- debug tournament?
- set up graphing/plotting
- set up neo4j

- optimize!
- make it easier to switch between weighted and pareto (and lexicographic) fitness vectors.
ideally, this is somethign you should be able to set in the config.toml

- make sure linear_gp works. probably a few kinks to work out. 

- implement a ROPush machine

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
  
- periodize diversity scores. every n epochs, copy the sketch to an immutable structure that's used for queries. Seasons. 

- TTL on genes. See appendix to thesis. how did this work?

- it would be interesting to try rewarding chains that read memory that they themselves have written. bit complex. might approximate by ignoring sequence. might not. all the info is available in the hatchery after all. 

    track addresses read from, and then later check intersection with addresses written to. abstract away from sequence. it's about building up useful material and control. 

- track and plot soup size

- PE files need support

- trial harness for experimental runs
- dockerize
- get it cloud ready
- island distribution


# Tomorrow:
## Graphing! We really need graphing. Top priority.
 
 