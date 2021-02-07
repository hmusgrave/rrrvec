# rrrvec

A simple RRR Vector implementation using SIMD in Cython and backed by Zig.

# Status
Unfinished, unmaintained. Snippets might selectively be useful to somebody curious how to:
- Use alternative compiler backends for Cython
- Use SIMD instructions in their Cython
- Abuse the AVX instruction set to get a limited-use vectorized 8-bit multiply

# Description

RRR Vectors are a compression technique which additionally provide
constant-time operations such as the rank (prefix-sum) up to a given bit.
They're an example of a succinct data structure.

The over-arching idea is to chunk your data and provide ancillary structures
supporting those compressed chunks which allow you to only have to decompress
the data you need, not the entire vector, in order to do your work.

As an example, suppose you want to store the chunk `0001000`. We break this into
two parts:
- The *popcount* is the number of 1s in your chunk. Since for a chunk of size 7
  we could have anywhere from 0-7 set bits we need 3 bits to encode the
  popcount and to be able to represent all possible values. In our case, this
  works out to be `001`.
- The *offset* is the index of your chunk in a 0-based sorted array containing
  every possible chunk with the same length and popcount. In our case the
  values 0000001, 0000010, and 0000100 are all smaller, so we're the 4th item,
  or *index 3*. Since there are 7 possible chunks which also have a popcount of
  1 we need at least 3 bits to uniquely represent all offsets. This translates
  to a representation of `100` for us.

Putting these two things together, `0001000` transforms to `001100` and saves a
single bit. Repeat for every chunk of data, and if it's locally sparse or dense you'll have a
smaller representation.

# Major deviations from standard implementations

When first introduced, RRR Vectors are described as a buffer full of
popcount/offset pairs and an associated superblock structure to enable skipping
past most of the elements. Operations are performed by using the superblock
index to hop to a superblock and linearly walking from one popcount/offset pair
to the next to get to the desired location.

That approach isn't particularly amenable to modern vectorized processors and
leaves a lot of performance on the table if the superblocks have any
substantial size (and if they don't then the superblock index can consume far
more space than intended in a succinct data structure). We make one small tweak
in the data layout to more easily enable vectorization. Since the popcounts all
have the same fixed width, place them adjacent to each other to enable a quick
scan and sum to find the desired offset location:

```
# Superblock
popcount | offset | popcount | offset | popcount | offset

# Modified Superblock
popcount | popcount | popcount | offset | offset | offset
```
