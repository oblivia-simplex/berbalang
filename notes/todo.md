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

- new fitness factor: bytes written. reinstate mem write hook.
- better: reward if one of the indirect values in the register pattern is written.
- reward bytes written to the extent that they approximate an indirect rp value