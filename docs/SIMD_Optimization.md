# **Performance Deep Dive: How we beat NumPy**

## **The Benchmark Story**

In Phase 2, we benchmarked our C++ Vector Database against NumPy.

### **Initial Results (Debug Mode)**

-   **NumPy:** ~0.014s
-   **C++ (Scalar):** ~0.048s
-   **Verdict:** C++ was **3x Slower**.

### **Optimized Results (Release Mode + AVX)**

-   **NumPy:** ~0.014s
-   **C++ (Vectorized):** ~0.0039s
-   **Verdict:** C++ became **3.8x Faster** than NumPy.

## **Why was NumPy faster initially?**

NumPy is not "Just Python." It is a wrapper around highly optimized C and Fortran libraries (BLAS/LAPACK).  
When you calculate distance in NumPy:  
diff = data - query

It doesn't subtract numbers one by one. It uses **SIMD (Single Instruction, Multiple Data)** instructions.

-   **Scalar (Your initial C++):** CPU handles 1 number at a time.
-   **SIMD (NumPy):** CPU handles 8 or 16 numbers at a time (using AVX2/AVX-512 registers).

Because our initial C++ compilation flags were generic, the compiler played it safe and generated "Scalar" assembly code.

## **The Fix: Compiler Flags**

We added these flags to `CMakeLists.txt`:

```
add_compile_options(-O3 -march=native -ffast-math)
```

1. **-O3 (Optimization Level 3):**
    - Tells the compiler: "Spend more time compiling to generate the most efficient code possible."
    - It enables loop unrolling and function inlining.
2. **-march=native (The Game Changer):**
    - Tells the compiler: "Look at the CPU inside _this specific computer_. Does it have AVX2? AVX-512? If yes, use them!"
    - This allows the compiler to rewrite our simple for loops into SIMD instructions automatically (**Auto-Vectorization**).
3. **-ffast-math:**
    - Allows the compiler to break strict IEEE-754 floating point rules for speed (e.g., assuming `(a*b)*c == a*(b*c)`, which isn't always true in float math but is fine for machine learning).

## **The "Secret Weapon": Memory Bandwidth**

Even with SIMD, why is C++ **faster** than NumPy (0.004s vs 0.014s)?

**The answer is Temporary Memory Allocation.**

## **How NumPy works:**

### 1. Allocates a HUGE temporary matrix (50,000 x 128 floats)

```
diff = data - query
```

### 2. Reads that huge matrix to calculate norm

```
dists = np.linalg.norm(diff)
```

NumPy creates a massive intermediate array in RAM. Writing to and reading from RAM is slow (The Von Neumann Bottleneck).

## **How our C++ works:**

```
for (int i = 0; i < N; ++i) {
   float dist = 0;
   for (int j = 0; j < 128; ++j) {
   // Calculation happens in CPU Registers (L1 Cache)
   // No temporary array is ever written to RAM
      dist += (data[i][j] - query[j]) * ...
   }
}
```

Our C++ implementation performs "Kernel Fusion." It does the subtraction and the squaring in a single pass without ever storing the intermediate subtraction results in main memory.

**Conclusion:** We won because we avoided the Memory Bandwidth bottleneck.
