# **Performance Deep Dive: CPU Cache Locality & Memory Layout**

## **The Problem: Summing Columns in a Matrix**

Imagine you have a large 2D Matrix (a grid of numbers), and you want to calculate the **Average of each Column**.

The Data: A Matrix with $N$ Rows and $M$ Columns.  
The Goal: Produce an array of size $M$ containing the sum/average of each vertical column.  
There are two ways to write the loops to achieve this. Mathematically, they are identical. Computationally, one is **drastically faster** due to how C++ handles memory.

## **1. Background: Row-Major Memory**

To understand performance, you must understand how C++ stores a `std::vector<std::vector<float>>` in RAM.

C++ uses **Row-Major** storage. This means:

1. **Row 0** is stored as a continuous block of numbers.
2. **Row 1** is stored as the next continuous block.
3. **Row 2** is next, and so on.

**Visualizing the RAM Addresses:**
Imagine a $3 \times 4$ Matrix (3 Rows, 4 Columns).

-   **Row 0:** Addresses 1000, 1004, 1008, 1012 (Continuous)
-   **Row 1:** Addresses 2000, 2004, 2008, 2012 (Continuous)
-   **Row 2:** Addresses 3000, 3004, 3008, 3012 (Continuous)

## **2. Approach A: Column-wise Traversal (The "Intuitive" Way)**

Since we want the sum of each column, it feels natural to freeze the column `j` and loop through all rows `i`.

### **The Code**

```
// BAD FOR PERFORMANCE
std::vector<float> sum_columns_bad(const std::vector<std::vector<float>>& matrix, int rows, int cols) {
 std::vector<float> result(cols, 0.0f);

    // Outer Loop: Iterate through COLUMNS (0, 1, 2...)
    for (int j = 0; j < cols; j++) {

        // Inner Loop: Iterate through ROWS
        for (int i = 0; i < rows; i++) {
            // We are jumping vertically through the matrix
            result[j] += matrix[i][j];
        }
    }
    return result;

}
```

### **The Performance Cost (Cache Misses)**

Let's trace the CPU's journey when calculating Column 0 (j=0):

1. **Read `matrix[0][0]`**: CPU goes to **Address 1000**.
    - _Side Effect:_ The CPU fetches a "Cache Line" (e.g., 64 bytes). It loads Addresses 1000, 1004, 1008, 1012 into the fast L1 Cache.
2. **Read `matrix[1][0]`**: CPU tries to read **Address 2000**.
    - _Problem:_ Address 2000 is NOT in the cache. The data loaded in step 1 is useless.
    - _Result:_ **Cache Miss.** The CPU stalls and waits to fetch data from slow RAM.
3. **Read `matrix[2][0]`**: CPU tries to read **Address 3000**.
    - _Result:_ **Cache Miss.**

**Verdict:** The CPU is constantly jumping ("striding") through memory. It discards useful data before it can use it.

## **3. Approach B: Row-wise Traversal (The "Fast" Way)**

Instead of jumping vertically, we process the matrix **horizontally** (Row by Row). We update the running totals for _all_ columns simultaneously.

### **The Code**

```
// GOOD FOR PERFORMANCE
std::vector<float> sum_columns_good(const std::vector<std::vector<float>>& matrix, int rows, int cols) {
 std::vector<float> result(cols, 0.0f);

    // Outer Loop: Iterate through ROWS (0, 1, 2...)
    for (int i = 0; i < rows; i++) {

        // Inner Loop: Iterate through COLUMNS
        // We read the entire Row 'i' in one smooth pass
        for (int j = 0; j < cols; j++) {
            result[j] += matrix[i][j];
        }
    }
    return result;

}
```

### **The Performance Gain (Spatial Locality)**

Let's trace the CPU's journey when processing Row 0 (i=0):

1. **Read `matrix[0][0]`**: CPU goes to **Address 1000**.
    - _Side Effect:_ CPU loads Addresses 1000, 1004, 1008, 1012 into L1 Cache.
2. **Read `matrix[0][1]`**: CPU asks for **Address 1004**.
    - _Result:_ **Cache Hit!** The data is already sitting in the cache from Step 1. Instant access.
3. **Read `matrix[0][2]`**: CPU asks for **Address 1008**.
    - _Result:_ **Cache Hit!**
4. **Read `matrix[0][3]`**: CPU asks for **Address 1012**.
    - _Result:_ **Cache Hit!**

**Verdict:** By accessing memory sequentially (Linear Access), we maximize **Spatial Locality**. The CPU spends almost 0 time waiting for RAM.

## **Summary**

| Feature            | Column-wise (Dim first)                        | Row-wise (Vector first)                  |
| :----------------- | :--------------------------------------------- | :--------------------------------------- |
| **Logic**          | Completes one column sum before moving to next | Accumulates partial sums for all columns |
| **Memory Access**  | Jumps (Strided)                                | Linear (Contiguous)                      |
| **CPU Cache**      | High Miss Rate (Slow)                          | High Hit Rate (Fast)                     |
| **Recommendation** | Avoid for large matrices                       | **Preferred for C++ Vectors**            |
