# The Effects of a Sketch-Based Novelty Metric on the "Minimize Zeroes" Problem

Before embarking on more ambitious tasks, it's prudent to first evaluate ROPER's ability to evolve
gadget chains that solve an extremely simple programming problem: find a payload that minimizes
the zero bits in the bitwise product of a small series of registers -- `EAX & EBX & ECX & EDX`, for
example. Our secondary goal here is to assess the impact of two different diversity pressures
on the evolutionary process:

1. A phenotypic "register novelty" metric, which assigns to each phenotype a score proportionate
   to the relative number of times that we have already seen the register state resulting from its
   execution. This will be measured using a 
   [count-min sketch](https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch)
   algorithm, which will occasionally (and unpredictably) mistake the novel for the known, but
   infrequently enough that it will be a small price to pay in exchange for
   the memory efficiency the algorithm affords -- requiring only a small, constant amount of space.
   
2. A developmental "gadget uniqueness" metric, which assigns to each phenotype a reward 
   proportionate to the number of distinct gadgets that it successfully executes. The idea here
   will be to incentivize the discovery of recombinable "building blocks", through which a
   genotype is able to maintain control over the flow of execution, in a sequential, decomposable
   fashion. 
   
## Populations

Our populations for this experiment will consist of 7 islands containing 768 genotypes each. Each
"seed" genotype will consist of between 10 and 100 64-bit integers. Six different configurations
will be used. In half of these, the initial "genetic soup" from which our genotypes will be
constructed will just be a set of 0x10_000 random, executable addresses, drawn from the target
binary. In the other, the "soup" will contain 0x18_cf4 gadgets harvested from the binary using 
the open source gadget discovery tool, [ROPGadget](https://github.com/JonathanSalwan/ROPgadget).
The target binary, in every case, will be a build of OpenSSH, version 6.8p1, statically compiled for
the x86 architecture. 

Beyond that, the populations are distinguished only by their fitness functions:

1. in one set of populations, the fitness function will be nothing but the number of zeroes in 
   the bitwise product of the registers `EAX & EBX & ECX & EDX`, which is to be minimized.
   The configuration files for this set can be found [here](./zeroes/random-soup-just-zeroes.toml)
   and [here](./zeroes/rg-soup-just-zeroes.toml);
2. in another set, this zero-bit count is taken together, in a weighted sum, with the
   ratio of phenotypes already witnessed to have *exactly* the same values in the four observed
   registers (EAX, EBX, ECX, and EDX), with a weight of 1000 -- which is, again, to be
   minimized. See the configuration files [here](./zeroes/random-soup-zeroes-novelty.toml)
   and [here](./zeroes/rg-soup-zeroes-novelty.toml);
3. in another, the fitness function combines these two addends with a third, set to
   `10 - min(10, x)`, where `x` is the number of distinct gadgets executed by a particular
   specimen: specimens are penalized for failing to execute 10 or more gadgets, but no
   additional reward is given for executing more than 10. Configuration files 
   [here](./zeroes/random-soup-zeroes-novelty-uniq.toml) and 
   [here](./zeroes/rg-soup-zeroes-novelty-uniq.toml).

## Selection Method

We use the [tournament selection](https://en.wikipedia.org/wiki/Tournament_selection) algorithm
to select parents for each new batch of offspring. A tournament size of six was used, throughout.
Using Lee Spector and John Klein's 
["trivial geography"](https://link.springer.com/chapter/10.1007/0-387-28111-8_8)
algorithm, the competitors for each tournament are all chosen from a 20-cell neighbourhood in
a one-dimensional population-space ("geography"), surrounding the first specimen chosen. 
The two best performers are selected as parents, while the two worst performers are excised from
the population, to be replaced by the parents' two offspring.

## Genetic Operators

Our genetic operators for this experiment include point mutation and an alternating crossover
technique.

### Mutation

Mutation is decided on a pointwise (allele-by-allele) basis, with decisions made by the
following function, so that the gaps between mutated alleles are drawn from an exponential
distribution: 

```rust
pub fn levy_decision<R: Rng>(rng: &mut R, length: usize, exponent: f64) -> bool {
    debug_assert!(length > 0);
    let thresh = 1.0 - (1.0 / length as f64);
    rand_distr::Exp::new(exponent)
        .expect("Bad exponent for Exp distribution")
        .sample(rng)
        >= thresh
}
```

`exponent`, here, has been set to 3.0.

The type of mutation is chosen with uniform probability from the following, where `v` is the
existing integer value of the allele:

1. `v` is interpreted as an address and dereferenced, if valid.
2. read-only memory is searched for `v`. If `v` is found at address `a`, we replace `v` with
   `a` in the genotype.
3. a random bit in `v` is flipped.
4. a number between 0 and 256 is chosen, with uniform probability, and added to `v`.
5. a number between 0 and 256 is chosen, with uniform probability, and subtracted from `v`.

### Crossover

Crossover between two genotypes is performed in the following fashion (in pseudocode,
omitting some bookkeeping that's not relevant here):

```$xslt
let D be an exponential distribution
let h be an array containing be two reading heads
let parent be an array containing the two parental genotypes

set h[p % 2] to the beginning of the first parent's genotype, and
set h[(p+1) % 2] to the beginning of the second parent's genotype

loop until we reach the end of a parental genotype:
    let h1' = h[p % 2] + a value drawn from D
    let h2' = h[(p+1) % 2] + a value drawn from D

    copy parent[p][h[p % 2]..h1'] to the child genotype
    set h[p % 2] <- h1'
    set h[(p+1) % 2] <- h2'

    increment p

then copy the rest of the remaining parental genotype to the child

return the child genotype
```

## Duration

These populations are then set to evolve for either 150 epochs, or until a "perfect" specimen
is produced, where "perfect", here, means that it was able to produce a register state where
```$xslt
EAX = 0xffff_ffff
EBX = 0xffff_ffff
ECX = 0xffff_ffff
EDX = 0xffff_ffff
```

An epoch is defined as the number of iterations it takes until every member of the population
is *expected* to have had a chance to reproduce: this is taken to be the size of the population,
divided by the tournament size.

## Migration

With every tournament, there's a 1 in 50 chance of migration between islands: a winner of the
most recent tournament is pushed to a "pier". If no such emigrant is selected, then we check
the pier to see if any immigrants are waiting. If they are, we add them to the population. 
No mechanism is in place to prevent an emigrant from immigrating back into its own native
island, from the pier, and so the effectiveness of this migration method, and the pattern
of migration that it induces, relies, to some extent, on the thread scheduler.

# Results

