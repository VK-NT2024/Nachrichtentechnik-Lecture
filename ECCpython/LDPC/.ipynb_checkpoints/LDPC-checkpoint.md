## LDPC

Low Density Parity Codes is a class of block codes with a sparse representation.

## Guide
the alist arrays should have the following format:
``` 
        [[12, 16],
        [4, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 8, 10, 13],
        [4, 7, 9, 13],
        [2, 5, 7, 10],
        [4, 6, 11, 14],
        [3, 9, 15, 16],
        [1, 6, 9, 10],
        [4, 8, 12, 15],
        [2, 6, 12, 16],
        [1, 7, 14, 16],
        [3, 5, 12, 14],
        [2, 11, 13, 15],
        [1, 5, 8, 11],
        [6, 9, 12],
        [3, 8, 11],
        [1, 5, 10],
        [2, 4, 7],
        [3, 10, 12],
        [4, 6, 8],
        [2, 3, 9],
        [1, 7, 12],
        [2, 5, 6],
        [1, 3, 6],
        [4, 11, 12],
        [7, 8, 10],
        [1, 2, 11],
        [4, 9, 10],
        [5, 7, 11],
        [5, 8, 9]]
```
AList files contain information about the parity-check matrix of the LDPC code.
Each row in the file represents a parity-check equation, and the columns represent the variable nodes.

## Usage
Given that u is defined with apporpriate length and Lin is the output of the channel:
```commandline
$ ldpc = LDPC(alist_file | alist_array)
$ ... # define u with size ldpc.K
$ c = ldpc.encode(u)
$ ... # define channel
$ Lc = ldpc.message_passing(Lin, iterations=10, algorithm="approximation")
$ c_hat = Lc[-1] <= 0
```

### Notes
1. The output of the message passing algorithm is an array that shows the Lc for each iteration.
The last row of Lc (obtained by Lc[-1]) is the final result.
2. softdecode performs the message passing but returns the final row only. This is
recommeneded for usage within a simulation loop.
3. You can see the generator using ldpc.G


## References
[1] http://www.inference.org.uk/mackay/codes/data.html