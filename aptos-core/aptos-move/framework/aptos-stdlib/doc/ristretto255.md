
<a name="0x1_ristretto255"></a>

# Module `0x1::ristretto255`

This module contains functions for Ristretto255 curve arithmetic, assuming addition as the group operation.

The order of the Ristretto255 elliptic curve group is $\ell = 2^252 + 27742317777372353535851937790883648493$, same
as the order of the prime-order subgroup of Curve25519.

This module provides two structs for encoding Ristretto elliptic curves to the developer:

- First, a 32-byte-sized CompressedRistretto struct, which is used to persist points in storage.

- Second, a larger, in-memory, RistrettoPoint struct, which is decompressable from a CompressedRistretto struct. This
larger struct can be used for fast arithmetic operations (additions, multiplications, etc.). The results can be saved
back into storage by compressing RistrettoPoint structs back to CompressedRistretto structs.

This module also provides a Scalar struct for persisting scalars in storage and doing fast arithmetic on them.

One invariant maintained by this module is that all CompressedRistretto structs store a canonically-encoded point,
which can always be decompressed into a valid point on the curve as a RistrettoPoint struct. Unfortunately, due to
limitations in our underlying curve25519-dalek elliptic curve library, this decompression will unnecessarily verify
the validity of the point and thus slightly decrease performance.

Similarly, all Scalar structs store a canonically-encoded scalar, which can always be safely operated on using
arithmetic operations.

In the future, we might support additional features:

* For scalars:
- batch_invert()

* For points:
- double()
+ The challenge is that curve25519-dalek does NOT export double for Ristretto points (nor for Edwards)

- double_and_compress_batch()

- fixed-base, variable-time via optional_mixed_multiscalar_mul() in VartimePrecomputedMultiscalarMul
+ This would require a storage-friendly RistrettoBasepointTable and an in-memory variant of it too
+ Similar to the CompressedRistretto and RistrettoPoint structs in this module
+ The challenge is that curve25519-dalek's RistrettoBasepointTable is not serializable


-  [Struct `Scalar`](#0x1_ristretto255_Scalar)
-  [Struct `CompressedRistretto`](#0x1_ristretto255_CompressedRistretto)
-  [Struct `RistrettoPoint`](#0x1_ristretto255_RistrettoPoint)
-  [Constants](#@Constants_0)
-  [Function `point_identity_compressed`](#0x1_ristretto255_point_identity_compressed)
-  [Function `point_identity`](#0x1_ristretto255_point_identity)
-  [Function `basepoint_compressed`](#0x1_ristretto255_basepoint_compressed)
-  [Function `basepoint`](#0x1_ristretto255_basepoint)
-  [Function `basepoint_mul`](#0x1_ristretto255_basepoint_mul)
-  [Function `new_compressed_point_from_bytes`](#0x1_ristretto255_new_compressed_point_from_bytes)
-  [Function `new_point_from_bytes`](#0x1_ristretto255_new_point_from_bytes)
-  [Function `new_point_from_sha512`](#0x1_ristretto255_new_point_from_sha512)
-  [Function `new_point_from_64_uniform_bytes`](#0x1_ristretto255_new_point_from_64_uniform_bytes)
-  [Function `point_decompress`](#0x1_ristretto255_point_decompress)
-  [Function `point_compress`](#0x1_ristretto255_point_compress)
-  [Function `point_to_bytes`](#0x1_ristretto255_point_to_bytes)
-  [Function `point_mul`](#0x1_ristretto255_point_mul)
-  [Function `point_mul_assign`](#0x1_ristretto255_point_mul_assign)
-  [Function `basepoint_double_mul`](#0x1_ristretto255_basepoint_double_mul)
-  [Function `point_add`](#0x1_ristretto255_point_add)
-  [Function `point_add_assign`](#0x1_ristretto255_point_add_assign)
-  [Function `point_sub`](#0x1_ristretto255_point_sub)
-  [Function `point_sub_assign`](#0x1_ristretto255_point_sub_assign)
-  [Function `point_neg`](#0x1_ristretto255_point_neg)
-  [Function `point_neg_assign`](#0x1_ristretto255_point_neg_assign)
-  [Function `point_equals`](#0x1_ristretto255_point_equals)
-  [Function `multi_scalar_mul`](#0x1_ristretto255_multi_scalar_mul)
-  [Function `new_scalar_from_bytes`](#0x1_ristretto255_new_scalar_from_bytes)
-  [Function `new_scalar_from_sha512`](#0x1_ristretto255_new_scalar_from_sha512)
-  [Function `new_scalar_from_u8`](#0x1_ristretto255_new_scalar_from_u8)
-  [Function `new_scalar_from_u64`](#0x1_ristretto255_new_scalar_from_u64)
-  [Function `new_scalar_from_u128`](#0x1_ristretto255_new_scalar_from_u128)
-  [Function `new_scalar_reduced_from_32_bytes`](#0x1_ristretto255_new_scalar_reduced_from_32_bytes)
-  [Function `new_scalar_uniform_from_64_bytes`](#0x1_ristretto255_new_scalar_uniform_from_64_bytes)
-  [Function `scalar_zero`](#0x1_ristretto255_scalar_zero)
-  [Function `scalar_is_zero`](#0x1_ristretto255_scalar_is_zero)
-  [Function `scalar_one`](#0x1_ristretto255_scalar_one)
-  [Function `scalar_is_one`](#0x1_ristretto255_scalar_is_one)
-  [Function `scalar_equals`](#0x1_ristretto255_scalar_equals)
-  [Function `scalar_invert`](#0x1_ristretto255_scalar_invert)
-  [Function `scalar_mul`](#0x1_ristretto255_scalar_mul)
-  [Function `scalar_mul_assign`](#0x1_ristretto255_scalar_mul_assign)
-  [Function `scalar_add`](#0x1_ristretto255_scalar_add)
-  [Function `scalar_add_assign`](#0x1_ristretto255_scalar_add_assign)
-  [Function `scalar_sub`](#0x1_ristretto255_scalar_sub)
-  [Function `scalar_sub_assign`](#0x1_ristretto255_scalar_sub_assign)
-  [Function `scalar_neg`](#0x1_ristretto255_scalar_neg)
-  [Function `scalar_neg_assign`](#0x1_ristretto255_scalar_neg_assign)
-  [Function `scalar_to_bytes`](#0x1_ristretto255_scalar_to_bytes)
-  [Function `new_point_from_sha512_internal`](#0x1_ristretto255_new_point_from_sha512_internal)
-  [Function `new_point_from_64_uniform_bytes_internal`](#0x1_ristretto255_new_point_from_64_uniform_bytes_internal)
-  [Function `point_is_canonical_internal`](#0x1_ristretto255_point_is_canonical_internal)
-  [Function `point_identity_internal`](#0x1_ristretto255_point_identity_internal)
-  [Function `point_decompress_internal`](#0x1_ristretto255_point_decompress_internal)
-  [Function `point_compress_internal`](#0x1_ristretto255_point_compress_internal)
-  [Function `point_mul_internal`](#0x1_ristretto255_point_mul_internal)
-  [Function `basepoint_mul_internal`](#0x1_ristretto255_basepoint_mul_internal)
-  [Function `basepoint_double_mul_internal`](#0x1_ristretto255_basepoint_double_mul_internal)
-  [Function `point_add_internal`](#0x1_ristretto255_point_add_internal)
-  [Function `point_sub_internal`](#0x1_ristretto255_point_sub_internal)
-  [Function `point_neg_internal`](#0x1_ristretto255_point_neg_internal)
-  [Function `multi_scalar_mul_internal`](#0x1_ristretto255_multi_scalar_mul_internal)
-  [Function `scalar_is_canonical_internal`](#0x1_ristretto255_scalar_is_canonical_internal)
-  [Function `scalar_from_u64_internal`](#0x1_ristretto255_scalar_from_u64_internal)
-  [Function `scalar_from_u128_internal`](#0x1_ristretto255_scalar_from_u128_internal)
-  [Function `scalar_reduced_from_32_bytes_internal`](#0x1_ristretto255_scalar_reduced_from_32_bytes_internal)
-  [Function `scalar_uniform_from_64_bytes_internal`](#0x1_ristretto255_scalar_uniform_from_64_bytes_internal)
-  [Function `scalar_invert_internal`](#0x1_ristretto255_scalar_invert_internal)
-  [Function `scalar_from_sha512_internal`](#0x1_ristretto255_scalar_from_sha512_internal)
-  [Function `scalar_mul_internal`](#0x1_ristretto255_scalar_mul_internal)
-  [Function `scalar_add_internal`](#0x1_ristretto255_scalar_add_internal)
-  [Function `scalar_sub_internal`](#0x1_ristretto255_scalar_sub_internal)
-  [Function `scalar_neg_internal`](#0x1_ristretto255_scalar_neg_internal)
-  [Specification](#@Specification_1)
    -  [Function `point_equals`](#@Specification_1_point_equals)
    -  [Function `new_point_from_sha512_internal`](#@Specification_1_new_point_from_sha512_internal)
    -  [Function `new_point_from_64_uniform_bytes_internal`](#@Specification_1_new_point_from_64_uniform_bytes_internal)
    -  [Function `point_is_canonical_internal`](#@Specification_1_point_is_canonical_internal)
    -  [Function `point_identity_internal`](#@Specification_1_point_identity_internal)
    -  [Function `point_decompress_internal`](#@Specification_1_point_decompress_internal)
    -  [Function `point_compress_internal`](#@Specification_1_point_compress_internal)
    -  [Function `point_mul_internal`](#@Specification_1_point_mul_internal)
    -  [Function `basepoint_mul_internal`](#@Specification_1_basepoint_mul_internal)
    -  [Function `basepoint_double_mul_internal`](#@Specification_1_basepoint_double_mul_internal)
    -  [Function `point_add_internal`](#@Specification_1_point_add_internal)
    -  [Function `point_sub_internal`](#@Specification_1_point_sub_internal)
    -  [Function `point_neg_internal`](#@Specification_1_point_neg_internal)
    -  [Function `multi_scalar_mul_internal`](#@Specification_1_multi_scalar_mul_internal)
    -  [Function `scalar_is_canonical_internal`](#@Specification_1_scalar_is_canonical_internal)
    -  [Function `scalar_from_u64_internal`](#@Specification_1_scalar_from_u64_internal)
    -  [Function `scalar_from_u128_internal`](#@Specification_1_scalar_from_u128_internal)
    -  [Function `scalar_reduced_from_32_bytes_internal`](#@Specification_1_scalar_reduced_from_32_bytes_internal)
    -  [Function `scalar_uniform_from_64_bytes_internal`](#@Specification_1_scalar_uniform_from_64_bytes_internal)
    -  [Function `scalar_invert_internal`](#@Specification_1_scalar_invert_internal)
    -  [Function `scalar_from_sha512_internal`](#@Specification_1_scalar_from_sha512_internal)
    -  [Function `scalar_mul_internal`](#@Specification_1_scalar_mul_internal)
    -  [Function `scalar_add_internal`](#@Specification_1_scalar_add_internal)
    -  [Function `scalar_sub_internal`](#@Specification_1_scalar_sub_internal)
    -  [Function `scalar_neg_internal`](#@Specification_1_scalar_neg_internal)


<pre><code><b>use</b> <a href="../../move-stdlib/doc/error.md#0x1_error">0x1::error</a>;
<b>use</b> <a href="../../move-stdlib/doc/option.md#0x1_option">0x1::option</a>;
<b>use</b> <a href="../../move-stdlib/doc/vector.md#0x1_vector">0x1::vector</a>;
</code></pre>



<a name="0x1_ristretto255_Scalar"></a>

## Struct `Scalar`

This struct represents a scalar as a little-endian byte encoding of an integer in $\mathbb{Z}_\ell$, which is
stored in <code>data</code>. Here, \ell denotes the order of the scalar field (and the underlying elliptic curve group).


<pre><code><b>struct</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> <b>has</b> <b>copy</b>, drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>data: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_ristretto255_CompressedRistretto"></a>

## Struct `CompressedRistretto`

This struct represents a serialized point on the Ristretto255 curve, in 32 bytes.
This struct can be decompressed from storage into an in-memory RistrettoPoint, on which fast curve arithmetic
can be performed.


<pre><code><b>struct</b> <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> <b>has</b> <b>copy</b>, drop, store
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>data: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="0x1_ristretto255_RistrettoPoint"></a>

## Struct `RistrettoPoint`

This struct represents an in-memory Ristretto255 point and supports fast curve arithmetic.

An important invariant: There will never be two RistrettoPoint's constructed with the same handle. One can have
immutable references to the same RistrettoPoint, of course.


<pre><code><b>struct</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> <b>has</b> drop
</code></pre>



<details>
<summary>Fields</summary>


<dl>
<dt>
<code>handle: u64</code>
</dt>
<dd>

</dd>
</dl>


</details>

<a name="@Constants_0"></a>

## Constants


<a name="0x1_ristretto255_A_PLUS_B_POINT"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_A_PLUS_B_POINT">A_PLUS_B_POINT</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [112, 207, 55, 83, 71, 91, 159, 243, 62, 47, 132, 65, 62, 214, 181, 5, 32, 115, 188, 204, 10, 10, 129, 120, 157, 62, 86, 117, 220, 37, 128, 86];
</code></pre>



<a name="0x1_ristretto255_A_PLUS_B_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_A_PLUS_B_SCALAR">A_PLUS_B_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [8, 56, 57, 221, 73, 30, 87, 197, 116, 55, 16, 195, 154, 145, 214, 229, 2, 202, 179, 207, 14, 39, 154, 228, 23, 217, 31, 242, 203, 99, 62, 7];
</code></pre>



<a name="0x1_ristretto255_A_POINT"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_A_POINT">A_POINT</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [232, 127, 237, 161, 153, 215, 43, 131, 222, 79, 91, 45, 69, 211, 72, 5, 197, 112, 25, 198, 197, 156, 66, 203, 112, 238, 61, 25, 170, 153, 111, 117];
</code></pre>



<a name="0x1_ristretto255_A_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_A_SCALAR">A_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [26, 14, 151, 138, 144, 246, 98, 45, 55, 71, 2, 63, 138, 216, 38, 77, 167, 88, 170, 27, 136, 224, 64, 209, 88, 158, 123, 127, 35, 118, 239, 9];
</code></pre>



<a name="0x1_ristretto255_A_TIMES_BASE_POINT"></a>

A_SCALAR * BASE_POINT, computed by modifying a test in curve25519-dalek in src/edwards.rs to do:
```
let comp = RistrettoPoint(A_TIMES_BASEPOINT.decompress().unwrap()).compress();
println!("hex: {:x?}", comp.to_bytes());
```


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_A_TIMES_BASE_POINT">A_TIMES_BASE_POINT</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [150, 213, 45, 146, 98, 238, 30, 26, 174, 121, 251, 174, 232, 193, 217, 6, 139, 13, 1, 191, 154, 69, 121, 230, 24, 9, 12, 61, 16, 136, 174, 16];
</code></pre>



<a name="0x1_ristretto255_A_TIMES_B_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_A_TIMES_B_SCALAR">A_TIMES_B_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [42, 181, 14, 56, 61, 124, 33, 15, 116, 213, 56, 115, 48, 115, 95, 24, 49, 81, 18, 209, 13, 251, 152, 252, 206, 30, 38, 32, 192, 192, 20, 2];
</code></pre>



<a name="0x1_ristretto255_BASE_POINT"></a>

The basepoint (generator) of the Ristretto255 group


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_BASE_POINT">BASE_POINT</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [226, 242, 174, 10, 106, 188, 78, 113, 168, 132, 169, 97, 197, 0, 81, 95, 88, 227, 11, 106, 165, 130, 221, 141, 182, 166, 89, 69, 224, 141, 45, 118];
</code></pre>



<a name="0x1_ristretto255_B_POINT"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_B_POINT">B_POINT</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [250, 11, 54, 36, 176, 129, 198, 47, 54, 77, 11, 40, 57, 220, 199, 109, 124, 58, 176, 226, 126, 49, 190, 178, 185, 237, 118, 101, 117, 242, 142, 118];
</code></pre>



<a name="0x1_ristretto255_B_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_B_SCALAR">B_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [219, 253, 151, 175, 211, 138, 6, 240, 19, 141, 5, 39, 239, 178, 142, 173, 91, 113, 9, 180, 134, 70, 89, 19, 191, 58, 164, 114, 168, 237, 78, 13];
</code></pre>



<a name="0x1_ristretto255_E_DIFFERENT_NUM_POINTS_AND_SCALARS"></a>

The number of scalars does not match the number of points.


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_E_DIFFERENT_NUM_POINTS_AND_SCALARS">E_DIFFERENT_NUM_POINTS_AND_SCALARS</a>: u64 = 1;
</code></pre>



<a name="0x1_ristretto255_E_ZERO_POINTS"></a>

Expected more than zero points as input.


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_E_ZERO_POINTS">E_ZERO_POINTS</a>: u64 = 2;
</code></pre>



<a name="0x1_ristretto255_E_ZERO_SCALARS"></a>

Expected more than zero scalars as input.


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_E_ZERO_SCALARS">E_ZERO_SCALARS</a>: u64 = 3;
</code></pre>



<a name="0x1_ristretto255_L_MINUS_ONE"></a>

<code><a href="ristretto255.md#0x1_ristretto255_ORDER_ELL">ORDER_ELL</a></code> - 1: i.e., the "largest", reduced scalar in the field


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_L_MINUS_ONE">L_MINUS_ONE</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [236, 211, 245, 92, 26, 99, 18, 88, 214, 156, 247, 162, 222, 249, 222, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16];
</code></pre>



<a name="0x1_ristretto255_L_PLUS_ONE"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_L_PLUS_ONE">L_PLUS_ONE</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [238, 211, 245, 92, 26, 99, 18, 88, 214, 156, 247, 162, 222, 249, 222, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16];
</code></pre>



<a name="0x1_ristretto255_L_PLUS_TWO"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_L_PLUS_TWO">L_PLUS_TWO</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [239, 211, 245, 92, 26, 99, 18, 88, 214, 156, 247, 162, 222, 249, 222, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16];
</code></pre>



<a name="0x1_ristretto255_MAX_POINT_NUM_BYTES"></a>

The maximum size in bytes of a canonically-encoded Ristretto255 point is 32 bytes.


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_MAX_POINT_NUM_BYTES">MAX_POINT_NUM_BYTES</a>: u64 = 32;
</code></pre>



<a name="0x1_ristretto255_MAX_SCALAR_NUM_BITS"></a>

The maximum size in bits of a canonically-encoded Scalar is 256 bits.


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_MAX_SCALAR_NUM_BITS">MAX_SCALAR_NUM_BITS</a>: u64 = 256;
</code></pre>



<a name="0x1_ristretto255_MAX_SCALAR_NUM_BYTES"></a>

The maximum size in bytes of a canonically-encoded Scalar is 32 bytes.


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_MAX_SCALAR_NUM_BYTES">MAX_SCALAR_NUM_BYTES</a>: u64 = 32;
</code></pre>



<a name="0x1_ristretto255_NON_CANONICAL_ALL_ONES"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_NON_CANONICAL_ALL_ONES">NON_CANONICAL_ALL_ONES</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
</code></pre>



<a name="0x1_ristretto255_ORDER_ELL"></a>

The order of the Ristretto255 group and its scalar field, in little-endian.


<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_ORDER_ELL">ORDER_ELL</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [237, 211, 245, 92, 26, 99, 18, 88, 214, 156, 247, 162, 222, 249, 222, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16];
</code></pre>



<a name="0x1_ristretto255_REDUCED_2_256_MINUS_1_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_REDUCED_2_256_MINUS_1_SCALAR">REDUCED_2_256_MINUS_1_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [28, 149, 152, 141, 116, 49, 236, 214, 112, 207, 125, 115, 244, 91, 239, 198, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 15];
</code></pre>



<a name="0x1_ristretto255_REDUCED_X_PLUS_2_TO_256_TIMES_X_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_REDUCED_X_PLUS_2_TO_256_TIMES_X_SCALAR">REDUCED_X_PLUS_2_TO_256_TIMES_X_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [216, 154, 179, 139, 210, 121, 2, 71, 69, 99, 158, 216, 23, 173, 63, 100, 204, 0, 91, 50, 219, 153, 57, 249, 28, 82, 31, 197, 100, 165, 192, 8];
</code></pre>



<a name="0x1_ristretto255_TWO_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_TWO_SCALAR">TWO_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
</code></pre>



<a name="0x1_ristretto255_X_INV_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_X_INV_SCALAR">X_INV_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [28, 220, 23, 252, 224, 233, 165, 187, 217, 36, 126, 86, 187, 1, 99, 71, 187, 186, 49, 237, 213, 169, 187, 150, 213, 11, 205, 122, 63, 150, 42, 15];
</code></pre>



<a name="0x1_ristretto255_X_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_X_SCALAR">X_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [78, 90, 180, 52, 93, 71, 8, 132, 89, 19, 180, 100, 27, 194, 125, 82, 82, 165, 133, 16, 27, 204, 66, 68, 212, 73, 244, 168, 121, 217, 242, 4];
</code></pre>



<a name="0x1_ristretto255_X_TIMES_Y_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_X_TIMES_Y_SCALAR">X_TIMES_Y_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [108, 51, 116, 161, 137, 79, 98, 33, 10, 170, 47, 225, 134, 166, 249, 44, 224, 170, 117, 194, 119, 149, 129, 194, 149, 252, 8, 23, 154, 115, 148, 12];
</code></pre>



<a name="0x1_ristretto255_Y_SCALAR"></a>



<pre><code><b>const</b> <a href="ristretto255.md#0x1_ristretto255_Y_SCALAR">Y_SCALAR</a>: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; = [144, 118, 51, 254, 28, 75, 102, 164, 162, 141, 45, 215, 103, 131, 134, 195, 83, 208, 222, 84, 85, 212, 252, 157, 232, 239, 122, 195, 31, 53, 187, 5];
</code></pre>



<a name="0x1_ristretto255_point_identity_compressed"></a>

## Function `point_identity_compressed`

Returns the identity point as a CompressedRistretto.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_identity_compressed">point_identity_compressed</a>(): <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">ristretto255::CompressedRistretto</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_identity_compressed">point_identity_compressed</a>(): <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> {
    <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> {
        data: x"0000000000000000000000000000000000000000000000000000000000000000"
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_identity"></a>

## Function `point_identity`

Returns the identity point as a CompressedRistretto.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_identity">point_identity</a>(): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_identity">point_identity</a>(): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_point_identity_internal">point_identity_internal</a>()
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_basepoint_compressed"></a>

## Function `basepoint_compressed`

Returns the basepoint (generator) of the Ristretto255 group as a compressed point


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_compressed">basepoint_compressed</a>(): <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">ristretto255::CompressedRistretto</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_compressed">basepoint_compressed</a>(): <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> {
    <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_BASE_POINT">BASE_POINT</a>
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_basepoint"></a>

## Function `basepoint`

Returns the basepoint (generator) of the Ristretto255 group


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint">basepoint</a>(): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint">basepoint</a>(): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <b>let</b> (handle, _) = <a href="ristretto255.md#0x1_ristretto255_point_decompress_internal">point_decompress_internal</a>(<a href="ristretto255.md#0x1_ristretto255_BASE_POINT">BASE_POINT</a>);

    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_basepoint_mul"></a>

## Function `basepoint_mul`

Multiplies the basepoint (generator) of the Ristretto255 group by a scalar and returns the result.
This call is much faster than <code><a href="ristretto255.md#0x1_ristretto255_point_mul">point_mul</a>(&<a href="ristretto255.md#0x1_ristretto255_basepoint">basepoint</a>(), &some_scalar)</code> because of precomputation tables.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_mul">basepoint_mul</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_mul">basepoint_mul</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_basepoint_mul_internal">basepoint_mul_internal</a>(a.data)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_compressed_point_from_bytes"></a>

## Function `new_compressed_point_from_bytes`

Creates a new CompressedRistretto point from a sequence of 32 bytes. If those bytes do not represent a valid
point, returns None.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_compressed_point_from_bytes">new_compressed_point_from_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;<a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">ristretto255::CompressedRistretto</a>&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_compressed_point_from_bytes">new_compressed_point_from_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): Option&lt;<a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a>&gt; {
    <b>if</b> (<a href="ristretto255.md#0x1_ristretto255_point_is_canonical_internal">point_is_canonical_internal</a>(bytes)) {
        std::option::some(<a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> {
            data: bytes
        })
    } <b>else</b> {
        std::option::none&lt;<a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a>&gt;()
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_point_from_bytes"></a>

## Function `new_point_from_bytes`

Creates a new RistrettoPoint from a sequence of 32 bytes. If those bytes do not represent a valid point,
returns None.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_bytes">new_point_from_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_bytes">new_point_from_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): Option&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>&gt; {
    <b>let</b> (handle, is_canonical) = <a href="ristretto255.md#0x1_ristretto255_point_decompress_internal">point_decompress_internal</a>(bytes);
    <b>if</b> (is_canonical) {
        std::option::some(<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> { handle })
    } <b>else</b> {
        std::option::none&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>&gt;()
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_point_from_sha512"></a>

## Function `new_point_from_sha512`

Hashes the input to a uniformly-at-random RistrettoPoint via SHA512.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_sha512">new_point_from_sha512</a>(sha512: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_sha512">new_point_from_sha512</a>(sha512: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_new_point_from_sha512_internal">new_point_from_sha512_internal</a>(sha512)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_point_from_64_uniform_bytes"></a>

## Function `new_point_from_64_uniform_bytes`

Samples a uniformly-at-random RistrettoPoint given a sequence of 64 uniformly-at-random bytes. This function
can be used to build a collision-resistant hash function that maps 64-byte messages to RistrettoPoint's.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_64_uniform_bytes">new_point_from_64_uniform_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_64_uniform_bytes">new_point_from_64_uniform_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): Option&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>&gt; {
    <b>if</b> (std::vector::length(&bytes) == 64) {
        std::option::some(<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
            handle: <a href="ristretto255.md#0x1_ristretto255_new_point_from_64_uniform_bytes_internal">new_point_from_64_uniform_bytes_internal</a>(bytes)
        })
    } <b>else</b> {
        std::option::none&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>&gt;()
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_decompress"></a>

## Function `point_decompress`

Decompresses a CompressedRistretto from storage into a RistrettoPoint which can be used for fast arithmetic.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_decompress">point_decompress</a>(point: &<a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">ristretto255::CompressedRistretto</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_decompress">point_decompress</a>(point: &<a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    // NOTE: Our <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> <b>invariant</b> assures us that every <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> in storage is a valid
    // <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>
    <b>let</b> (handle, _) = <a href="ristretto255.md#0x1_ristretto255_point_decompress_internal">point_decompress_internal</a>(point.data);
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> { handle }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_compress"></a>

## Function `point_compress`

Compresses a RistrettoPoint to a CompressedRistretto which can be put in storage.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_compress">point_compress</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">ristretto255::CompressedRistretto</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_compress">point_compress</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> {
    <a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_point_compress_internal">point_compress_internal</a>(point)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_to_bytes"></a>

## Function `point_to_bytes`

Returns the sequence of bytes representin this Ristretto point.
To convert a RistrettoPoint 'p' to bytes, first compress it via <code>c = <a href="ristretto255.md#0x1_ristretto255_point_compress">point_compress</a>(&p)</code>, and then call this
function on <code>c</code>.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_to_bytes">point_to_bytes</a>(point: &<a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">ristretto255::CompressedRistretto</a>): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_to_bytes">point_to_bytes</a>(point: &<a href="ristretto255.md#0x1_ristretto255_CompressedRistretto">CompressedRistretto</a>): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; {
    point.data
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_mul"></a>

## Function `point_mul`

Returns a * point.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_mul">point_mul</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_mul">point_mul</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_point_mul_internal">point_mul_internal</a>(point, a.data, <b>false</b>)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_mul_assign"></a>

## Function `point_mul_assign`

Sets a *= point and returns 'a'.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_mul_assign">point_mul_assign</a>(point: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_mul_assign">point_mul_assign</a>(point: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_point_mul_internal">point_mul_internal</a>(point, a.data, <b>true</b>);
    point
}
</code></pre>



</details>

<a name="0x1_ristretto255_basepoint_double_mul"></a>

## Function `basepoint_double_mul`

Returns (a * some_point + b * base_point), where base_point is the Ristretto basepoint encoded in <code><a href="ristretto255.md#0x1_ristretto255_BASE_POINT">BASE_POINT</a></code>.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_double_mul">basepoint_double_mul</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>, some_point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_double_mul">basepoint_double_mul</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>, some_point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_basepoint_double_mul_internal">basepoint_double_mul_internal</a>(a.data, some_point, b.data)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_add"></a>

## Function `point_add`

Returns a + b


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_add">point_add</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_add">point_add</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_point_add_internal">point_add_internal</a>(a, b, <b>false</b>)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_add_assign"></a>

## Function `point_add_assign`

Sets a += b and returns 'a'.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_add_assign">point_add_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_add_assign">point_add_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_point_add_internal">point_add_internal</a>(a, b, <b>true</b>);
    a
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_sub"></a>

## Function `point_sub`

Returns a - b


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_sub">point_sub</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_sub">point_sub</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_point_sub_internal">point_sub_internal</a>(a, b, <b>false</b>)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_sub_assign"></a>

## Function `point_sub_assign`

Sets a -= b and returns 'a'.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_sub_assign">point_sub_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_sub_assign">point_sub_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_point_sub_internal">point_sub_internal</a>(a, b, <b>true</b>);
    a
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_neg"></a>

## Function `point_neg`

Returns -a


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_neg">point_neg</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_neg">point_neg</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_point_neg_internal">point_neg_internal</a>(a, <b>false</b>)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_neg_assign"></a>

## Function `point_neg_assign`

Sets a = -a, and returns 'a'.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_neg_assign">point_neg_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_neg_assign">point_neg_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <a href="ristretto255.md#0x1_ristretto255_point_neg_internal">point_neg_internal</a>(a, <b>true</b>);
    a
}
</code></pre>



</details>

<a name="0x1_ristretto255_point_equals"></a>

## Function `point_equals`

Returns true if the two RistrettoPoints are the same points on the elliptic curve.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_equals">point_equals</a>(g: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, h: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_equals">point_equals</a>(g: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, h: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): bool;
</code></pre>



</details>

<a name="0x1_ristretto255_multi_scalar_mul"></a>

## Function `multi_scalar_mul`

Computes a multi-scalar multiplication, returning a_1 p_1 + a_2 p_2 + ... + a_n p_n.
This function is much faster than computing each a_i p_i using <code>point_mul</code> and adding up the results using <code>point_add</code>.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_multi_scalar_mul">multi_scalar_mul</a>(points: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>&gt;, scalars: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>&gt;): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_multi_scalar_mul">multi_scalar_mul</a>(points: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>&gt;, scalars: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>&gt;): <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
    <b>assert</b>!(!std::vector::is_empty(points), std::error::invalid_argument(<a href="ristretto255.md#0x1_ristretto255_E_ZERO_POINTS">E_ZERO_POINTS</a>));
    <b>assert</b>!(!std::vector::is_empty(scalars), std::error::invalid_argument(<a href="ristretto255.md#0x1_ristretto255_E_ZERO_SCALARS">E_ZERO_SCALARS</a>));
    <b>assert</b>!(std::vector::length(points) == std::vector::length(scalars), std::error::invalid_argument(<a href="ristretto255.md#0x1_ristretto255_E_DIFFERENT_NUM_POINTS_AND_SCALARS">E_DIFFERENT_NUM_POINTS_AND_SCALARS</a>));

    <a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a> {
        handle: <a href="ristretto255.md#0x1_ristretto255_multi_scalar_mul_internal">multi_scalar_mul_internal</a>&lt;<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>&gt;(points, scalars)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_scalar_from_bytes"></a>

## Function `new_scalar_from_bytes`

Given a sequence of 32 bytes, checks if they canonically-encode a Scalar and return it.
Otherwise, returns None.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_bytes">new_scalar_from_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_bytes">new_scalar_from_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): Option&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>&gt; {
    <b>if</b> (<a href="ristretto255.md#0x1_ristretto255_scalar_is_canonical_internal">scalar_is_canonical_internal</a>(bytes)) {
        std::option::some(<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
            data: bytes
        })
    } <b>else</b> {
        std::option::none&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>&gt;()
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_scalar_from_sha512"></a>

## Function `new_scalar_from_sha512`

Hashes the input to a uniformly-at-random Scalar via SHA512


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_sha512">new_scalar_from_sha512</a>(sha512_input: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_sha512">new_scalar_from_sha512</a>(sha512_input: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_scalar_from_sha512_internal">scalar_from_sha512_internal</a>(sha512_input)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_scalar_from_u8"></a>

## Function `new_scalar_from_u8`

Creates a Scalar from an u8.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_u8">new_scalar_from_u8</a>(byte: u8): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_u8">new_scalar_from_u8</a>(byte: u8): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <b>let</b> s = <a href="ristretto255.md#0x1_ristretto255_scalar_zero">scalar_zero</a>();
    <b>let</b> byte_zero = std::vector::borrow_mut(&<b>mut</b> s.data, 0);
    *byte_zero = byte;

    s
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_scalar_from_u64"></a>

## Function `new_scalar_from_u64`

Creates a Scalar from an u64.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_u64">new_scalar_from_u64</a>(eight_bytes: u64): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_u64">new_scalar_from_u64</a>(eight_bytes: u64): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_scalar_from_u64_internal">scalar_from_u64_internal</a>(eight_bytes)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_scalar_from_u128"></a>

## Function `new_scalar_from_u128`

Creates a Scalar from an u128.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_u128">new_scalar_from_u128</a>(sixteen_bytes: u128): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_from_u128">new_scalar_from_u128</a>(sixteen_bytes: u128): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_scalar_from_u128_internal">scalar_from_u128_internal</a>(sixteen_bytes)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_scalar_reduced_from_32_bytes"></a>

## Function `new_scalar_reduced_from_32_bytes`

Creates a Scalar from 32 bytes by reducing the little-endian-encoded number in those bytes modulo $\ell$.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_reduced_from_32_bytes">new_scalar_reduced_from_32_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_reduced_from_32_bytes">new_scalar_reduced_from_32_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): Option&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>&gt; {
    <b>if</b> (std::vector::length(&bytes) == 32) {
        std::option::some(<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
            data: <a href="ristretto255.md#0x1_ristretto255_scalar_reduced_from_32_bytes_internal">scalar_reduced_from_32_bytes_internal</a>(bytes)
        })
    } <b>else</b> {
        std::option::none()
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_scalar_uniform_from_64_bytes"></a>

## Function `new_scalar_uniform_from_64_bytes`

Samples a scalar uniformly-at-random given 64 uniform-at-random bytes as input by reducing the little-endian-encoded number
in those bytes modulo $\ell$.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_uniform_from_64_bytes">new_scalar_uniform_from_64_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_scalar_uniform_from_64_bytes">new_scalar_uniform_from_64_bytes</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): Option&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>&gt; {
    <b>if</b> (std::vector::length(&bytes) == 64) {
        std::option::some(<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
            data: <a href="ristretto255.md#0x1_ristretto255_scalar_uniform_from_64_bytes_internal">scalar_uniform_from_64_bytes_internal</a>(bytes)
        })
    } <b>else</b> {
        std::option::none()
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_zero"></a>

## Function `scalar_zero`

Returns 0 as a Scalar.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_zero">scalar_zero</a>(): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_zero">scalar_zero</a>(): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: x"0000000000000000000000000000000000000000000000000000000000000000"
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_is_zero"></a>

## Function `scalar_is_zero`

Returns true if the given Scalar equals 0.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_is_zero">scalar_is_zero</a>(s: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_is_zero">scalar_is_zero</a>(s: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): bool {
    s.data == x"0000000000000000000000000000000000000000000000000000000000000000"
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_one"></a>

## Function `scalar_one`

Returns 1 as a Scalar.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_one">scalar_one</a>(): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_one">scalar_one</a>(): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: x"0100000000000000000000000000000000000000000000000000000000000000"
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_is_one"></a>

## Function `scalar_is_one`

Returns true if the given Scalar equals 1.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_is_one">scalar_is_one</a>(s: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_is_one">scalar_is_one</a>(s: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): bool {
    s.data == x"0100000000000000000000000000000000000000000000000000000000000000"
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_equals"></a>

## Function `scalar_equals`

Returns true if the two scalars are equal.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_equals">scalar_equals</a>(lhs: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>, rhs: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_equals">scalar_equals</a>(lhs: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>, rhs: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): bool {
    lhs.data == rhs.data
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_invert"></a>

## Function `scalar_invert`

Returns the inverse s^{-1} mod \ell of a scalar s.
Returns None if s is zero.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_invert">scalar_invert</a>(s: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="../../move-stdlib/doc/option.md#0x1_option_Option">option::Option</a>&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_invert">scalar_invert</a>(s: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): Option&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>&gt; {
    <b>if</b> (<a href="ristretto255.md#0x1_ristretto255_scalar_is_zero">scalar_is_zero</a>(s)) {
        std::option::none&lt;<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>&gt;()
    } <b>else</b> {
        std::option::some(<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
            data: <a href="ristretto255.md#0x1_ristretto255_scalar_invert_internal">scalar_invert_internal</a>(s.data)
        })
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_mul"></a>

## Function `scalar_mul`

Returns the product of the two scalars.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_mul">scalar_mul</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_mul">scalar_mul</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_scalar_mul_internal">scalar_mul_internal</a>(a.data, b.data)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_mul_assign"></a>

## Function `scalar_mul_assign`

Computes the product of 'a' and 'b' and assigns the result to 'a'.
Returns 'a'.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_mul_assign">scalar_mul_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_mul_assign">scalar_mul_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    a.data = <a href="ristretto255.md#0x1_ristretto255_scalar_mul">scalar_mul</a>(a, b).data;
    a
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_add"></a>

## Function `scalar_add`

Returns the sum of the two scalars.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_add">scalar_add</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_add">scalar_add</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_scalar_add_internal">scalar_add_internal</a>(a.data, b.data)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_add_assign"></a>

## Function `scalar_add_assign`

Computes the sum of 'a' and 'b' and assigns the result to 'a'
Returns 'a'.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_add_assign">scalar_add_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_add_assign">scalar_add_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    a.data = <a href="ristretto255.md#0x1_ristretto255_scalar_add">scalar_add</a>(a, b).data;
    a
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_sub"></a>

## Function `scalar_sub`

Returns the difference of the two scalars.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_sub">scalar_sub</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_sub">scalar_sub</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_scalar_sub_internal">scalar_sub_internal</a>(a.data, b.data)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_sub_assign"></a>

## Function `scalar_sub_assign`

Subtracts 'b' from 'a' and assigns the result to 'a'.
Returns 'a'.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_sub_assign">scalar_sub_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_sub_assign">scalar_sub_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>, b: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    a.data = <a href="ristretto255.md#0x1_ristretto255_scalar_sub">scalar_sub</a>(a, b).data;
    a
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_neg"></a>

## Function `scalar_neg`

Returns the negation of 'a': i.e., $(0 - a) \mod \ell$.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_neg">scalar_neg</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_neg">scalar_neg</a>(a: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
        data: <a href="ristretto255.md#0x1_ristretto255_scalar_neg_internal">scalar_neg_internal</a>(a.data)
    }
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_neg_assign"></a>

## Function `scalar_neg_assign`

Replaces 'a' by its negation.
Returns 'a'.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_neg_assign">scalar_neg_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_neg_assign">scalar_neg_assign</a>(a: &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): &<b>mut</b> <a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a> {
    a.data = <a href="ristretto255.md#0x1_ristretto255_scalar_neg">scalar_neg</a>(a).data;
    a
}
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_to_bytes"></a>

## Function `scalar_to_bytes`

Returns the byte-representation of the scalar.


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_to_bytes">scalar_to_bytes</a>(s: &<a href="ristretto255.md#0x1_ristretto255_Scalar">ristretto255::Scalar</a>): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_to_bytes">scalar_to_bytes</a>(s: &<a href="ristretto255.md#0x1_ristretto255_Scalar">Scalar</a>): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt; {
    s.data
}
</code></pre>



</details>

<a name="0x1_ristretto255_new_point_from_sha512_internal"></a>

## Function `new_point_from_sha512_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_sha512_internal">new_point_from_sha512_internal</a>(sha512: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_sha512_internal">new_point_from_sha512_internal</a>(sha512: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_new_point_from_64_uniform_bytes_internal"></a>

## Function `new_point_from_64_uniform_bytes_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_64_uniform_bytes_internal">new_point_from_64_uniform_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_64_uniform_bytes_internal">new_point_from_64_uniform_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_point_is_canonical_internal"></a>

## Function `point_is_canonical_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_is_canonical_internal">point_is_canonical_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_is_canonical_internal">point_is_canonical_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): bool;
</code></pre>



</details>

<a name="0x1_ristretto255_point_identity_internal"></a>

## Function `point_identity_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_identity_internal">point_identity_internal</a>(): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_identity_internal">point_identity_internal</a>(): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_point_decompress_internal"></a>

## Function `point_decompress_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_decompress_internal">point_decompress_internal</a>(maybe_non_canonical_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): (u64, bool)
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_decompress_internal">point_decompress_internal</a>(maybe_non_canonical_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): (u64, bool);
</code></pre>



</details>

<a name="0x1_ristretto255_point_compress_internal"></a>

## Function `point_compress_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_compress_internal">point_compress_internal</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_compress_internal">point_compress_internal</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_point_mul_internal"></a>

## Function `point_mul_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_mul_internal">point_mul_internal</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, in_place: bool): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_mul_internal">point_mul_internal</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, in_place: bool): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_basepoint_mul_internal"></a>

## Function `basepoint_mul_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_mul_internal">basepoint_mul_internal</a>(a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_mul_internal">basepoint_mul_internal</a>(a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_basepoint_double_mul_internal"></a>

## Function `basepoint_double_mul_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_double_mul_internal">basepoint_double_mul_internal</a>(a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, some_point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_double_mul_internal">basepoint_double_mul_internal</a>(a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, some_point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, b: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_point_add_internal"></a>

## Function `point_add_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_add_internal">point_add_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, in_place: bool): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_add_internal">point_add_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, in_place: bool): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_point_sub_internal"></a>

## Function `point_sub_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_sub_internal">point_sub_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, in_place: bool): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_sub_internal">point_sub_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, in_place: bool): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_point_neg_internal"></a>

## Function `point_neg_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_neg_internal">point_neg_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, in_place: bool): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_neg_internal">point_neg_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">RistrettoPoint</a>, in_place: bool): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_multi_scalar_mul_internal"></a>

## Function `multi_scalar_mul_internal`

The generic arguments are needed to deal with some Move VM peculiarities which prevent us from borrowing the
points (or scalars) inside a &vector in Rust.

WARNING: This function can only be called with P = RistrettoPoint and S = Scalar.


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_multi_scalar_mul_internal">multi_scalar_mul_internal</a>&lt;P, S&gt;(points: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;P&gt;, scalars: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;S&gt;): u64
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_multi_scalar_mul_internal">multi_scalar_mul_internal</a>&lt;P, S&gt;(points: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;P&gt;, scalars: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;S&gt;): u64;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_is_canonical_internal"></a>

## Function `scalar_is_canonical_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_is_canonical_internal">scalar_is_canonical_internal</a>(s: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): bool
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_is_canonical_internal">scalar_is_canonical_internal</a>(s: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): bool;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_from_u64_internal"></a>

## Function `scalar_from_u64_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_u64_internal">scalar_from_u64_internal</a>(num: u64): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_u64_internal">scalar_from_u64_internal</a>(num: u64): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_from_u128_internal"></a>

## Function `scalar_from_u128_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_u128_internal">scalar_from_u128_internal</a>(num: u128): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_u128_internal">scalar_from_u128_internal</a>(num: u128): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_reduced_from_32_bytes_internal"></a>

## Function `scalar_reduced_from_32_bytes_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_reduced_from_32_bytes_internal">scalar_reduced_from_32_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_reduced_from_32_bytes_internal">scalar_reduced_from_32_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_uniform_from_64_bytes_internal"></a>

## Function `scalar_uniform_from_64_bytes_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_uniform_from_64_bytes_internal">scalar_uniform_from_64_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_uniform_from_64_bytes_internal">scalar_uniform_from_64_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_invert_internal"></a>

## Function `scalar_invert_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_invert_internal">scalar_invert_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_invert_internal">scalar_invert_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_from_sha512_internal"></a>

## Function `scalar_from_sha512_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_sha512_internal">scalar_from_sha512_internal</a>(sha512_input: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_sha512_internal">scalar_from_sha512_internal</a>(sha512_input: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_mul_internal"></a>

## Function `scalar_mul_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_mul_internal">scalar_mul_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_mul_internal">scalar_mul_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_add_internal"></a>

## Function `scalar_add_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_add_internal">scalar_add_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_add_internal">scalar_add_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_sub_internal"></a>

## Function `scalar_sub_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_sub_internal">scalar_sub_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_sub_internal">scalar_sub_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="0x1_ristretto255_scalar_neg_internal"></a>

## Function `scalar_neg_internal`



<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_neg_internal">scalar_neg_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>



<details>
<summary>Implementation</summary>


<pre><code><b>native</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_neg_internal">scalar_neg_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;;
</code></pre>



</details>

<a name="@Specification_1"></a>

## Specification


<a name="@Specification_1_point_equals"></a>

### Function `point_equals`


<pre><code><b>public</b> <b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_equals">point_equals</a>(g: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, h: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): bool
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_new_point_from_sha512_internal"></a>

### Function `new_point_from_sha512_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_sha512_internal">new_point_from_sha512_internal</a>(sha512: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_new_point_from_64_uniform_bytes_internal"></a>

### Function `new_point_from_64_uniform_bytes_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_new_point_from_64_uniform_bytes_internal">new_point_from_64_uniform_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_point_is_canonical_internal"></a>

### Function `point_is_canonical_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_is_canonical_internal">point_is_canonical_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): bool
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_point_identity_internal"></a>

### Function `point_identity_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_identity_internal">point_identity_internal</a>(): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_point_decompress_internal"></a>

### Function `point_decompress_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_decompress_internal">point_decompress_internal</a>(maybe_non_canonical_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): (u64, bool)
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_point_compress_internal"></a>

### Function `point_compress_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_compress_internal">point_compress_internal</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_point_mul_internal"></a>

### Function `point_mul_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_mul_internal">point_mul_internal</a>(point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, in_place: bool): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_basepoint_mul_internal"></a>

### Function `basepoint_mul_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_mul_internal">basepoint_mul_internal</a>(a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_basepoint_double_mul_internal"></a>

### Function `basepoint_double_mul_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_basepoint_double_mul_internal">basepoint_double_mul_internal</a>(a: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, some_point: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_point_add_internal"></a>

### Function `point_add_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_add_internal">point_add_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, in_place: bool): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_point_sub_internal"></a>

### Function `point_sub_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_sub_internal">point_sub_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, b: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, in_place: bool): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_point_neg_internal"></a>

### Function `point_neg_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_point_neg_internal">point_neg_internal</a>(a: &<a href="ristretto255.md#0x1_ristretto255_RistrettoPoint">ristretto255::RistrettoPoint</a>, in_place: bool): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_multi_scalar_mul_internal"></a>

### Function `multi_scalar_mul_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_multi_scalar_mul_internal">multi_scalar_mul_internal</a>&lt;P, S&gt;(points: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;P&gt;, scalars: &<a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;S&gt;): u64
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_is_canonical_internal"></a>

### Function `scalar_is_canonical_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_is_canonical_internal">scalar_is_canonical_internal</a>(s: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): bool
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_from_u64_internal"></a>

### Function `scalar_from_u64_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_u64_internal">scalar_from_u64_internal</a>(num: u64): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_from_u128_internal"></a>

### Function `scalar_from_u128_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_u128_internal">scalar_from_u128_internal</a>(num: u128): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_reduced_from_32_bytes_internal"></a>

### Function `scalar_reduced_from_32_bytes_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_reduced_from_32_bytes_internal">scalar_reduced_from_32_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_uniform_from_64_bytes_internal"></a>

### Function `scalar_uniform_from_64_bytes_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_uniform_from_64_bytes_internal">scalar_uniform_from_64_bytes_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_invert_internal"></a>

### Function `scalar_invert_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_invert_internal">scalar_invert_internal</a>(bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_from_sha512_internal"></a>

### Function `scalar_from_sha512_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_from_sha512_internal">scalar_from_sha512_internal</a>(sha512_input: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_mul_internal"></a>

### Function `scalar_mul_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_mul_internal">scalar_mul_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_add_internal"></a>

### Function `scalar_add_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_add_internal">scalar_add_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_sub_internal"></a>

### Function `scalar_sub_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_sub_internal">scalar_sub_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;, b_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>



<a name="@Specification_1_scalar_neg_internal"></a>

### Function `scalar_neg_internal`


<pre><code><b>fun</b> <a href="ristretto255.md#0x1_ristretto255_scalar_neg_internal">scalar_neg_internal</a>(a_bytes: <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;): <a href="../../move-stdlib/doc/vector.md#0x1_vector">vector</a>&lt;u8&gt;
</code></pre>




<pre><code><b>pragma</b> opaque;
</code></pre>


[move-book]: https://move-language.github.io/move/introduction.html
