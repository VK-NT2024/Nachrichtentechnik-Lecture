## Tools

Provides puncturing and interleaving of codes.

## Classes

2. Puncturing: provides punturing and depuncturing
2. Interleaving: provides interleavers to perform interleaving and deinterleaving

## Puncturing Guide

Given a puncture matrix, a puncturing class is defined that can be used to perform puncturing
and depuncturing efficiently. If the code object is provided, the code rate will be changed
as puncturing changed the rate.

```commandline
from Tools.puncturing import Puncturing

spc3 = SPC(3)
print(spc3.Rc) # 3/4
puncture_matrix = [[1, 1, 0], [ 1, 0, 0]]
punc = Puncturing(puncture_matrix = puncture_matrix, code=spc3)
print(spc3.Rc) # 3/4 * 6/3 = 3/2
```
Next codess from SPC(3) class can be punctured and depunctured as follows:

```commandline
c_punc = punc.puncture(u)
c_depunc = punc.depuncture(c_punc, unpunctured_size=None)
```

## Interleaving Guide

Interleaving is essentially a shuffling of the code symbols, and each defined interleaver works
essentially by defining a reshuffling index. The following interleavers are supported:

1. random: indices are shuffled (pseudo) randomly
2. block: indices are shuffled via transpose
3. s-random: indices are shuffled via a practical pseudo random generator
4. None: no index shuffling

```commandline
from Tools.interleaving import Interleaving

# to define a block interleaver of shape (3, 4)
block = Interleaving(interleaver_type="block",
                 shape=(3, 4),
                 seed=None,
                 S_value=None)

# to define a random interleaver of shape (12,)            
rand = Interleaving(interleaver_type="random",
                 shape=(12,),
                 seed=42,
                 S_value=None)
```
important: The shape argument needs to be a tuple. To perform interleaving and deinterleaving,
it is then a simple process. The input and output will always have the same dimensions.


```commandline
u_interleaved = block.interleave(u)
u_deinterleaved = block.deinterleave(u_interleaved)
```