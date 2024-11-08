## Mapping
This class is responsible for mapping and modulation.

## Classes
1. Mapping: Defines gray, antigray, natural, random mappings
2. Modulation: Defines QAM, PSK, ASK

## Guide
Modulation inherits from Mapping since the creation of a modulation scheme is dependent on the
mapping scheme. The different mapping schemes will therefore change how the constellation
diagram looks. The constellation can be called as follows:

```commandline
from Mapping.modulation import Modulation

mod = Modulation(m=2 , coding_type="gray", modulation_type="QAM")
mod.contellation
```
The class can perform modulation, hard and soft demodulation and simulation.

```commandline
x = mod.modulate(u)
y = x + w # output from AWGN channel
Lin = Lch * y # Lch is channel reliability
u_hat = mod.demodulate(y) # hard demodulation
Lu, Lc = mod.softdemodulation(Lin, La=None, algorithm="approximation")
```
## References