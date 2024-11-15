## Modulation
This class is responsible for mapping and modulation.

## Classes
1. Mapping: Defines gray, antigray, natural, random mappings
2. Modulation: Defines QAM, PSK, ASK, modulation and demodulation routines, and filter

## Guide
Modulation inherits from Mapping since the creation of a modulation scheme is dependent on the
mapping scheme. The different mapping schemes will therefore change how the constellation
diagram looks. The constellation can be called as follows:

```commandline
from Modulation.modulation import Modulation

mod = Modulation(m=2 , coding_type="gray", modulation_type="QAM")
mod.contellation
```
The class can perform modulation, hard and soft demodulation and simulation. 

```commandline
x = mod.modulate(u)
y = x + w # output from AWGN channel
u_hat = mod.demodulate(y) # hard demodulation
Lin = Lch * y # Lch is channel reliability
Lu, Lc = mod.softdemodulation(Lin, La=None, algorithm="approximation")
```

It also includes various transmit filters like
* rectangular filter
* triangular filter
* root raised cosine filter
* raised cosine filter
* gaussian filter
* GMSK filter

The method 'impulse_shaping()' performs upsampling of the sequence of data symbols and the convolution with the 
filters impulse response.

```commandline
m = 4
ask = Modulation(m, coding_type='gray',modulation_type='ASK')

# generate impouse response of root raised cosine and time vector
Ts = 1e-6   # symbol duration in seconds
w = 8       # oversampling factor
fa = w / Ts # sampling rate in Hz
N_rc = 8*w  # length of filter impulse response in samples
r = 0.25    # rolloff factor of root raised cosine filter
 
time, g_rrc = ask.generate_g_rrc(N_rc,r,Ts,fa)

# perform convolution of data sequence with impulse response
time2,x_rrc = ask.impulse_shaping(d_ask, 'rrc', N_rc, Ts, fa, r)
```


## References