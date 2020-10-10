cdef extern from "x86intrin.h":
    ctypedef unsigned int __m256i
ctypedef __m256i i256

cdef extern from "x86intrin.h":
    i256 loadu256 '_mm256_loadu_si256' (const i256 *from_addr) nogil
    i256 broad32 '_mm256_set1_epi32' (int a) nogil
    i256 mask256 '_mm256_and_si256' (i256 dat, i256 mask) nogil
    i256 sub8 '_mm256_sub_epi8' (i256 a, i256 b) nogil
    i256 maddu16 '_mm256_maddubs_epi16' (i256 a, i256 b) nogil
    i256 muli32 '_mm256_mullo_epi32' (i256 a, i256 b) nogil
    i256 add8 '_mm256_add_epi8' (i256 a, i256 b) nogil
    i256 rshift16 '_mm256_srai_epi16' (i256 a, int count) nogil
    i256 cmpeq8 '_mm256_cmpeq_epi8' (i256 a, i256 b) nogil
    i256 andnot256 '_mm256_andnot_si256' (i256 a, i256 b) nogil
    void storeu256 '_mm256_storeu_si256' (i256 *to_addr, i256 dat) nogil

cdef void pvec(str label, i256 vec):
    """TODO: remove or move to utils when project is finished"""
    cdef unsigned int[8] out
    storeu256(<i256*>out, vec)
    def mhex(i):
        s = f'{i:x}'
        return '0'*(8-len(s)) + s
    print(label, *map(mhex, out), sep=', ')

cdef i256 sum_widths_from_popcounts(i256 x, i256 y):
    '''
    Computes a vector of 8-bit ints which when added yield the total compressed width
    represented by x and y.

    Returns:
        32x 8-bit ints in range(0x00, 0x34 + 1)

    TODO:
        Ought to be merged into a method which maskloads blocks, operates on
        more 256-bit vectors at once, and finishes with a horizontal summation.
        Leaving as is till that is finished because this is already tested.

    Summary:
        Each 4-bit lane in x or y represents the popcount of a 15-bit block
        being compressed. Each block is represented by its popcount and its
        offset if all 15-bit blocks with that popcount were sorted (0-indexed),
        but the challenge is that we're using as few bits as possible, so to
        find the offset we care about we need to compute how far to jump.

        All popcounts are adjacent, so we can mask and sum (conveniently, the
        width for popcount 0 is also 0, so a blind masking procedure gives the
        correct result for popcounts we wish not to consider).

        The table we're trying to compute in each 4-bit lane to compute
        sum(map(f, arr)) is symmetric about 7.5. The first thing we do is map
        the 8+ values to their symmetric counterparts so that maddubs() will
        behave as needed because we can ensure none of the signed arguments
        are negative.

         x | f(x)
        =========
         0 |  0
         1 |  4
         2 |  7
         3 |  9
         4 | 11
         5 | 12
         6 | 13
         7 | 13
         8 | 13
         9 | 13
        10 | 12
        11 | 11
        12 | 9
        13 | 7
        14 | 4
        15 | 0

        Once the hi->lo mapping is finished, we apply an integer parabolic fit
        to the table (using maddubs() and a careful masking to achieve 8-bit
        multiplies) and correct the only broken value at 0. Add up the 4
        intermediate vectors so that we can return a single value.

        Instructions are interleaved to help out the compiler since most of the
        SIMD instructions we're using are high-latency. We alternate between
        processing x and y, and for the bulk of the work being done each of x
        and y is also split into two deep chains of independent operations, so
        with few exceptions there are 4 independent operations going down the
        pipeline.

        Otherwise, no major performance tuning has been done.
    '''

    # constants
    cdef i256 m41, s47, m47, m81, m84, l14, l15, r14, r15, sm
    m41, m47 = broad32(0x11111111), broad32(0x77777777)
    m81, m84 = broad32(0x01010101), broad32(0x04040404)
    l14, l15 = broad32(0xe00ee00e), broad32(0xf00ff00f)
    r14, r15 = broad32(0x0ee00ee0), broad32(0x0ff00ff0)
    s47, sm = broad32(0x00000007), broad32(0xff3fff3f)

    # intermediate data
    cdef i256 a, b, c, d
    cdef i256 e, f, g, h

    # map 4-bit lanes greater than 7 to their lesser symmetric counterparts
    a, e = rshift16(x, 3), rshift16(y, 3)
    a, e = mask256(a, m41), mask256(e, m41)
    a, e = muli32(a, s47), muli32(e, s47)
    b, f = sub8(m47, a), sub8(m47, e)
    a, e = andnot256(x, a), andnot256(y, e)
    b, f = mask256(b, x), mask256(f, y)
    x, y = add8(a, b), add8(e, f)
    
    # convert to widths
    # 
    # if copy-pasting note that the 0x0ff0-masked lanes will
    # have their widths reversed. we don't care because we're
    # just adding them up.
    a, e = mask256(x, l15), mask256(y, l15)
    c, g = mask256(x, r15), mask256(y, r15)
    b, f = sub8(l14, a), sub8(l14, e)
    d, h = sub8(r14, c), sub8(r14, g)
    a, e = maddu16(b, a), maddu16(f, e)
    c, g = maddu16(d, c), maddu16(h, g)
    a, e = add8(a, m84), add8(e, m84)
    c, g = add8(c, m84), add8(g, m84)
    a, e = rshift16(a, 2), rshift16(e, 2)
    c, g = rshift16(c, 2), rshift16(g, 2)
    a, e = mask256(a, sm), mask256(e, sm)
    c, g = mask256(c, sm), mask256(g, sm)
    b, f = cmpeq8(a, m81), cmpeq8(e, m81)
    d, h = cmpeq8(c, m81), cmpeq8(g, m81)
    b, f = mask256(b, m81), mask256(f, m81)
    d, h = mask256(d, m81), mask256(h, m81)
    a, e = sub8(a, b), sub8(e, f)
    c, g = sub8(c, d), sub8(g, h)

    return add8(add8(a, c), add8(e, g))

cdef void print_before_after(i256 dat):
    pvec(' input', dat)
    cdef i256 res = sum_widths_from_popcounts(dat, dat)
    pvec('output', res)

def run(x):
    """Unpacks x to an __m256i vector and performs some currently meaningless
    AVX operations"""
    cdef unsigned int[8] dat
    dat[:] = [0]*8
    for i in range(min(8, len(x))):
        dat[i] = x[i]
    print_before_after(loadu256(<i256*>dat))
