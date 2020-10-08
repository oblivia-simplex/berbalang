# Notes

Things to discuss:
- the Push VM implementation, and its observed effects (so far unremarkable).
- geography
- compositionality. This should be its own section.
- Describe the experiments, and sketch out the discussions, pending results. -- Code coverage -- Memory patterns -- Register patterns
- crossover

# On the Development of Berbalang (ROPER II)

## A General Framework for Genetic Programming Experimentation

## Modifications to the Unicorn Emulation Library and its Rust Bindings

When we finally scaled our experiments to more powerful AWS servers, and began performing longer runs, we started seeing occasional (though infrequent) segmentation faults. We dumped the core, and saw that these faults were only being triggered when Unicorn's timeout watchdog called the `uc_emu_stop()` function. 


![pwndbg session](./img/unicorn_segfault.png)


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


### Asexual reproduction (cloning + mutation) vs. Crossover

In "A Mixability Theory for the Role of Sex in Evolution," Adi Livnat et al. ask what selective pressures might account for the ubiquity of sexual reproduction in nature:

> We develop a measure, [mixability], which represents the genome-wide ability of alleles
> to perform well across different combinations. Using numerical iterations within a
> classical population-genetic framework, we find that sex favors the increase in 
> [mixability] in a highly robust manner. Furthermore, we expose the mechanism underlying
> this effect and find that it operates during the evolutionary transient, which has been
> studied relatively little. We also find that the breaking down of highly favourable
> gene combinations is an integral part of this mechanism. Therefore, if the roles of
> sex involves selection not for the best combinations of genes, as would be registered
> by [fitness], but for genes that are favourable in many different combinations, as is 
> registered by [mixability], then the breaking down of highly favourable combinations 
> does not necessarily pose a problem. 

We expect that the domain of ROP chain evolution might prove to be an interesting case by which to test Livnat's theory, particularly given that the evolution of ROP chains from a soup of random addresses places the problem of composability and mixability front and centre. In traditional genetic programming environments, the composability of instructions is more or less assured *a priori*. Here, by contrast, maintaining control over the flow of execution is an achievement to be won. [[flesh out a bit more.f uck i'm tired today]]

A simple, somewhat crude measure of how composable the alleles circulating in a population are can be found in the number of return instructions each specimen executes on average, since these mark the points at which various strings of alleles can be composed. (This measure can be deceived by specimens which create return-loops for themselves, whereby a gadget pushes its own address onto the stack before executing `ret`. But there is no prima facie reason to expect looping behaviour to be more common in sexual populations than asexual ones.)

TODO: we should also perform post-mortem analyses of mixability, using the metric explained in the paper. get the average fitness of every specimen containing an *executed* copy of the allele. BUT consider this: an allele that solves the problem in one stroke is highly mixable by this definition. This isn't a bug with the definition, really, but it should affect how we think of it as "playing well with others". If we didn't make the changes we made to the way execution traces are committed, then this property would describe many of our crashing local optima traps. 


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

