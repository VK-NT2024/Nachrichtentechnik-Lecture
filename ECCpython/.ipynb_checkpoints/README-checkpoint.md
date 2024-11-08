# ECC

Error Control Coding repository. Purpose is the simulation of
error correcting codes for the purpose of research and teaching.

Error-correcting codes are used to reliably transmit digital 
data over unreliable communication channels subject to channel
noise. When a sender wants to transmit a possibly very long 
data stream using a block code, the sender breaks the stream 
up into pieces of some fixed size. Each such piece is called 
message and the procedure given by the block code encodes each
message individually into a codeword, also called a block in 
the context of block codes. The sender then transmits all 
blocks to the receiver, who can in turn use some decoding 
mechanism to (hopefully) recover the original messages from 
the possibly corrupted received blocks. The performance and 
success of the overall transmission depends on the parameters 
of the channel and the block code. [1]

## Classes

Currently, the following classes are supported:

1. Block Codes
   1. Single Parity Check
   2. Repetition Code
   3. Hamming Code
2. Convolutional Codes
3. LDPC
4. Modulation
5. EXIT
6. Turbo Codes (not fully implemented)

## Examples

As an example for usage, if we want to define a SPC 3 code,
we need:

```commandline
$ from BlockCodes.spc import SPC
$ spc_3 = SPC(k=3)
```

This allows us to perform encoding and decoding and run
simulations.

Additionally, we can perform 16-QAM modulation using:

```commandline
$ from Mapping.modulation import Modulation 
$ mod = Modulation(m=4, coding_type="gray", modulation_type="QAM")
```

The usage of Turbo Codes and EXIT classes follows a slightly
different paradigm, namely: The object initialisation is done
using objects of other code classes. For example:

```commandline
$ from EXIT.exit import EXIT
$ ex = EXIT(spc_3, mod)
```

This then allows us to plot EXIT charts for BICM (bit 
interleaved coded modulation).

## References:
[1] https://en.wikipedia.org/wiki/Block_code