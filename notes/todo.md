# TODO

- Basic block bigram diversity measure using a count min sketch.

- debug tournament?
- set up graphing/plotting
- set up neo4j

- optimize!
- make it easier to switch between weighted and pareto fitness vectors.
ideally, this is somethign you should be able to set in the config.toml

- make sure linear_gp works. probably a few kinks to work out. 

- implement a ROPush machine

- NOTE: gadget count can be distorted when gadgets overlap. find a better
way of counting executed gadgets. maybe by scanning the basic block list.

e.g: suppose the chromosome contains [0x1000, 0x1001, 0x1003], but
0x1001 and 0x1003 belong to the same basic block as 0x1000. Then they
should not count as distinct gadgets. 

How to check this?
- iterate over basic block list, filtering against gadget set.

- new fitness factor: bytes written. reinstate mem write hook.
- better: reward if one of the indirect values in the register pattern is written.
- reward bytes written to the extent that they approximate an indirect rp value