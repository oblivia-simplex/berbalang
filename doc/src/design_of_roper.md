# Notes

Things to discuss:
- the Push VM implementation, and its observed effects (so far unremarkable).
- geography
- compositionality. This should be its own section.
- Describe the experiments, and sketch out the discussions, pending results. -- Code coverage -- Memory patterns -- Register patterns
- crossover

# Experiments

In this stage of our investigation, we set up three distinct fitness functions, to gauge the different ways in which ROPER's evolving payloads might exercise control over a target process.

TODO: For each, explain what the motivating idea is, what our hypotheses are.

## Variants

### Blind (random) search

It's a fairly simple matter to implement a more or less "blind" or random search by forcing the fitness function to return a constant value (1.0, for example), so that it becomes impossible to decide between n randomly-chosen tournament contestants on the basis of fitness. Sorting them in order of fitness, when fitness is always equal, will just preserve the random order in which their lots are drawn. 

### Initializing with harvested gadgets

Is the selective pressure imposed by any particular fitness function sufficient to steer the population towards the discovery of composable "gadgets"? What might happen if, rather than seeding the population with the addresses of gadgets, harvested in advance by a tool such as ROPGadget, we seeded it with randomly chosen addresses, without paying any attention to whether or not those addresses carve the executable at its return-oriented joints?

It seems reasonable to expect somewhat poorer performance in populations that are compelled to winnow out a stockpile of composable gadgets on their own. But to what degree? How difficult is the auxiliary task of gadget discovery, as compared to the objectives explicitly assigned by each fitness function?

### Rewarding behavioural novelty

A common pitfall in all forms of evolutionary search is the premature convergence of the population into merely local optima. An apparently simple way to steer the population

### Disallowing function calls
 
One particular place where our evolutionary processes appeared to get stuck was in cases where the problem was *almost entirely* solved by a single function call (possibly involving several nested calls). Such partial solutions could be prematurely discovered, and genetic strains containing them could swiftly dominate the gene pool, pushing the evolutionary process towards convergence before a good stock of generally useful materials could be produced. Even a "perfect" solution obtained by such means is of limited interest to us. Problems that can be solved in such a way rarely warrant the approach of return-oriented programming -- a simple return-to-libc attack is then sufficient. And since such solutions can only be discovered in a single, lucky stroke, their availability effectively interfered with the very phenomenon we wished to study: the discovery of composable "instructions" in a weird machine's instruction set, and the gradual composition of those instructions into programmatic solutions. 


### One-point vs. alternating crossover

The crossover algorithm we begin with had the feature of strongly limiting variation in chromosome length -- a child would, with rare exceptions, have a length equal to that of one of its two parents'. This does a good job of controlling code bloat, but required us to predict the best average length for payloads in advance, or to discover it through painstaking trial and error. 

We recently implemented a second, somewhat simpler crossover algorithm ("one-point crossover") which allows for a freer variation of chromosome lengths.


### Trivial Geography

Another technique we implemented, in an effort to modulate the spread of successful genes through the population and inhibit premature convergence, was to outfit the population structures with one-dimensional, or "trivial", geographic constraints. [cite Spector et al] The way this works

## Register Pattern Matching (Argument Preparation for System Calls)

The object of this task is to set a subset of registers to certain immediate or referenced values, such as we might wish to do if we were preparing to execute a particular system call. If our goal is to dispatch a call to `execve("/bin/sh", NULL, NULL)`, on the x86, for instance, we would want to have `EAX` set to `11`, the code for the `execve` system call, `EBX` set to some pointer to the string `"/bin/sh"`, `ECX` set to some pointer to `0`, and `EDX` set to `0`. One way of expressing this target is to treat it as a 4-dimensional surface in an n-dimensional space, where n is the total number of registers under observation. At a first approximation, we can express proximity to this register pattern as the *distance* between any point in that space and that surface.

This approach stumbles over several complicating factors, however, which account for the observed difficulty of the problem. [[Look at some of the literature on fitness landscapes here.]]


## Memory Writes

## Code Coverage



## Compositionality

What sets ROPER apart from other genetic programming environments, more than anything else, is that the composability of the instructions that comprise its genetic material is a feat to achieve and not a given.

We made a significant change to the system, in this iteration, in order to better foster the evolutionary discovery of composable components, and avoid certain dead-ends and sticky local optima. Whereas, previously, we recorded the execution behaviour of each specimen on an instruction-by-instruction basis, from the beginning of the chain's execution up until its termination (whether by crash, interrupt, or arrival at the address 0), we now maintain a temporary execution log that is commited to the specimen's execution profile *only* when a return instruction is reached -- which is to say, when another address is about to be popped from the attacker-controlled stack into the instruction pointer. We further restrict this "commitment points" to return instructions that are executed when the call stack is empty, and, in some runs, explore the option of halting execution when *any* function call is dispatched, so that every return is a return to a potentially (perhaps indirectly) attacker-controlled address. (This method could be generalized to treat jumps to addresses in stack-controlled registers as commitment points, too, with a bit of tinkering.)

Any instructions that are executed without eventually reaching such a "commitment point", for all intents and purposes, leave no trace. This is crucial. A sequence of instructions that partially, or even fully, satisfies one of our objectives, but which then crashes, or times out in an endless loop, is of no use to the population, because it cannot be *composed* with other sequences.

## TODO: what happens when we use PSHAPE to generate the soup? Can we?

## Experiments with reusing soup, filtered by some sort of frequency threshold
perhaps.

