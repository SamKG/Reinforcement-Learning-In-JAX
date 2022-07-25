# What is this?
This is an implementation of various RL tasks using [JAX](https://github.com/google/jax) + [RLax](https://github.com/deepmind/rlax). I also use the wonderful library [Equinox](https://github.com/patrick-kidger/equinox) to make code more readible.

# Why?
Here I list some high-level benefits of this approach, in no particular order:
1. **Performance**: JAX JIT compiles code for maximum performance. The goal of this repository is to explore how performant JAX could be for reinforcement learning.
2. **Readbility**: JAX + Equinox is (in my opinion) significantly more readible than implementations in PyTorch or other frameworks. The usage of functional patterns (e.g. `jax.vmap`) lends itself well to building mental models of code, as each unit of code is built for the simplest case (i.e. a single iteration), and parallelized. To maximize readability, I also avoid reshaping tensors wherever possible - everything should be obvious from a cursory read through the code. JAX's PyTrees are also incredibly convenient to work with.
3. **Reproducibility**: JAX has no surprises or magic. Everything is stateless, and hence less time is spent hunting through Python's classes for any state which could change results. Moreover, random seeds (i.e. PRNG Keys) are explicit in code. This makes it exceptionally easy to reproduce the same experiment.

There are certainly some tradeoffs to using JAX as well:

1. **Early stages**: JAX is still extremely early in development. There's a lot of moving parts, and the API or other functionality could change at any point. Moreover, setting up JAX can sometimes be a bit annoying - I had to manually export the path to CUDA-11.7 as JAX wouldn't find it automatically. Moreover, it's sometimes not obvious why performance is poor for certain segments of code.
2. **Compability**: JAX relies heavily on JIT compilation for performant execution - however, JIT compilation functionality currently depends on using only pure functions. Notably, this means that environments such as OpenAI Gym will not benefit from JIT compilation to the same extent as a pure version of gym. 


