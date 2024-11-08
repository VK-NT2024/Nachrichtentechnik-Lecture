## Convolutional Codes

A convolutional code is a type of error-correcting code that generates parity symbols 
via the sliding application of a boolean polynomial function to a data stream. The sliding
application represents the 'convolution' of the encoder over the data, which gives rise 
to the term 'convolutional coding'. The sliding nature of the convolutional codes 
facilitates trellis decoding using a time-invariant trellis. Time invariant trellis 
decoding allows convolutional codes to be maximum-likelihood soft-decision decoded with 
reasonable complexity. [1]

## Classes

1. Trellis: Initializes the trellis for the ConvCode object.
2. ConvCode: provides encoding, decoding and simulation of convolution codes

## Guide

The first step is importing the class, then initializing an object:
```commandline
from ConvolutionCodes.convcode import ConvCode

CC75 = ConvCode(
            generator_poly = [[1,1,1],[1,0,1],
            rsc_poly = 1,
            terminated = True
            )
```
In this example a (7, 5) convolution code is initialized. The generator_poly is short-
hand for the polynomaial written in binary, where each row describes the coefficients
of the polynomial. rsc_poly functions as a pointer to produce RSC code, this deletes 
one of the rows pointed to by the rsc_poly. Here rsc_poly=1 deletes the first row.
The Terminated field defines whether the trellis is going to be forced to reach the 
zero state.

Next, we look at the basic operations of encoding and decoding:
```commandline
u = np.random.randint(2, size=100)
c = CC75.encode(u) # This creates the coded word 
```
There are several methods that have been defined for decoding:
First, the Viterbiti algorithm:

The algorithm has found universal application in decoding the convolutional codes used 
in both CDMA and GSM digital cellular, dial-up modems, satellite, deep-space 
communications, and 802.11 wireless LANs. It is now also commonly used in speech 
recognition, speech synthesis, diarization, keyword spotting, computational 
linguistics, and bioinformatics. For example, in speech-to-text (speech recognition), 
the acoustic signal is treated as the observed sequence of events, and a string of text
is considered to be the "hidden cause" of the acoustic signal. The Viterbi algorithm 
finds the most likely string of text given the acoustic signal.[2]

```commandline
u_hat = CC75.viterbi_decode(c, decision_depth=None)
```
This performs the viterbi decoding. Note that the decision_depth can be used to perform
the sliding window viterbi algorithm (not fully tested)

Next, we have the BCJR algorithm(s) which are based on maximum likelihood estimation.
The input to the BCJR decoder is the LLR of the channel. Optionally the Apriori LLR can
be input as well, for turbo decoding. There are three algorithms available under BCJR:

1. BCJR (exact)
2. MaxLogMap (approximation)
3. LogMap (Exact calculation of LLR in log domain)

```commandline
Lin = Lch*y # y should be the output from the channel and Lch is a constant that depends on SNR
u_hat = CC75.softdecode(Lin, La=None, algorithm="maxlogmap")
```
The algorithm field takes "BCJR", "maxlogmap" or "logmap". Additionally, "exact" and 
"approximation" can be used. Note that the default decoder is "maxlogmap", and using the
method while leaving the algorithm field empty, will use the default decoder. The default
decoder can be changed as follows:

```commandline
CC75.default = "BCJR" # default is changed to the exact BCJR
```

This can be useful while working with EXIT or Turbo classes.

Finally, the class also supports a simulation method, that automatically sets up an AWGN channel
and simulates the performance of the code object. This has been implemented for the softdecoder.

```commandline
CC75.simulate(
            info_length=100, 
            trials=100, 
            SNR_min=0, 
            SNR_max=10, 
            SNR_step=1, 
            algorithm="maxlogmap")
```
And returns two plots: Probability of error vs Symbol Error Rate and Probability of error vs 
Bit Error Rate. Note that this doesn't take puncturing into consideration, and for punctured
codes the axis will not work correctly. A work-around can be to edit the Rc field manually.

## References
[1] https://en.wikipedia.org/wiki/Convolutional_code

[2] https://en.wikipedia.org/wiki/Viterbi_algorithm