# BV-SQ: Bias-Variance Scalar Quantization for Non-uniform Codebook Construction

## 0. Mục tiêu

Ghi chú này trình bày một phương pháp xây dựng codebook non-uniform scalar quantization theo từng row/output channel, lấy cảm hứng từ SqueezeLLM-style 1D scalar quantization nhưng thay objective Fisher-only bằng một objective bias-variance gần hơn với output-error loss.

Mục tiêu là có một solver:

```text
sort weights
→ compute prefix sums
→ find interval boundaries
→ compute codewords
```

Tức là chạy từ đầu như `sqllm`, không cần khởi tạo bằng một codebook có sẵn, không cần full Hessian dense, không cần iterative LNQ.

Phương pháp được gọi tạm là:

```text
BV-SQ: Bias-Variance Scalar Quantization
```

Trong đó:

- `BV` = bias-variance
- `SQ` = scalar quantization

Ghi chú này trình bày:

1. Bài toán scalar non-uniform quantization.
2. Loss bias-variance diagonal.
3. Công thức closed-form cho codeword tối ưu của một interval.
4. Prefix sums để tính interval cost trong `O(1)`.
5. Hai cách giải nhanh:
   - `BV-HierSplit`: two-split hierarchical quantization.
   - `BV-GreedySplit`: greedy best-split quantization.
6. Pseudo-code đủ rõ để code lại.
7. Các lưu ý implementation.

---

## 1. Bối cảnh: từ SqueezeLLM-style scalar quantization đến BV-SQ

Với một weight row/output channel:

\[
w = [w_1, w_2, \ldots, w_n]
\]

và target bitwidth `b`, số codeword là:

\[
K = 2^b.
\]

Scalar non-uniform quantization muốn thay mỗi weight bằng một codeword:

\[
\hat w_i = c_{a_i}, \quad a_i \in \{1,\ldots,K\}.
\]

SqueezeLLM-style weighted scalar quantization thường tối ưu dạng:

\[
\min_{\{a_i\},\{c_k\}}
\sum_i s_i (w_i - c_{a_i})^2,
\]

trong đó `s_i` là sensitivity/Fisher/Hessian-diagonal weight.

Vì weight là scalar 1D, ta sort:

\[
x_1 \le x_2 \le \cdots \le x_n.
\]

Sau khi sort, một nghiệm scalar quantization tự nhiên có dạng các đoạn liên tiếp:

\[
I_1=[1,t_1],\quad
I_2=[t_1+1,t_2],\quad\ldots,\quad
I_K=[t_{K-1}+1,n].
\]

Do đó bài toán trở thành tìm boundary giữa các interval.

BV-SQ giữ cấu trúc này, nhưng thay cost của một interval bằng cost có cả:

1. **Variance/reconstruction term**: diagonal Hessian/Fisher weighted error.
2. **Bias/mean-shift term**: phạt mean-shift của output do activation mean khác 0.

---

## 2. Bias-variance motivation

Xét một linear layer với output channel đang xét. Gọi residual quantization là:

\[
e = \hat w - w.
\]

Với activation input vector `x`, output error là:

\[
x^\top e.
\]

Đặt:

\[
\mu = \mathbb{E}[x],
\qquad
\Sigma = \operatorname{Cov}(x).
\]

Expected squared output error là:

\[
\mathbb{E}[(x^\top e)^2]
= (\mu^\top e)^2 + e^\top \Sigma e.
\]

Trong đó:

- \((\mu^\top e)^2\) là **bias / first-moment shift**.
- \(e^\top\Sigma e\) là **variance / covariance-weighted reconstruction error**.

Full covariance hoặc full Hessian sẽ khó tối ưu trực tiếp khi xây codebook, vì các off-diagonal terms làm các weight tương tác với nhau:

\[
e^\top \Sigma e
= \sum_i \Sigma_{ii} e_i^2
+ 2\sum_{i<j}\Sigma_{ij}e_i e_j.
\]

BV-SQ dùng diagonal approximation để giữ được cấu trúc interval:

\[
\Sigma \approx \operatorname{diag}(h_1,\ldots,h_n).
\]

Khi đó variance term là:

\[
\sum_i h_i e_i^2.
\]

---

## 3. Local interval bias-variance loss

Nếu dùng full channel bias:

\[
(\mu^\top e)^2
= \left(\sum_i \mu_i e_i\right)^2,
\]

thì các interval bị coupled với nhau. Điều này làm mất cấu trúc additive theo interval.

Để có solver nhanh kiểu `sqllm`, BV-SQ dùng **local interval bias**:

\[
\sum_{k=1}^K
\left(
\sum_{i\in I_k} \mu_i e_i
\right)^2.
\]

Với một interval \([a,b]\), nếu toàn bộ điểm trong interval được lượng tử hóa về cùng codeword `c`, residual là:

\[
e_i = c - x_i.
\]

Interval loss được định nghĩa là:

\[
\operatorname{Cost}(a,b)
=
\min_c
\left[
\lambda_b
\left(
\sum_{i=a}^b \mu_i(c-x_i)
\right)^2
+
\sum_{i=a}^b h_i(c-x_i)^2
\right].
\]

Trong ghi chú này, mặc định:

\[
\boxed{\lambda_b = 1.}
\]

Nghĩa là tham số cho mean shift / bias term mặc định bằng `1`.

Trong code, có thể đặt tên:

```python
bias_lambda = 1.0
```

Nếu không truyền tham số, dùng mặc định `bias_lambda=1.0`.

---

## 4. Closed-form codeword cho một interval

Với interval \([a,b]\), định nghĩa các tổng:

\[
M = \sum_{i=a}^b \mu_i,
\]

\[
U = \sum_{i=a}^b \mu_i x_i,
\]

\[
A = \sum_{i=a}^b h_i,
\]

\[
B = \sum_{i=a}^b h_i x_i,
\]

\[
C = \sum_{i=a}^b h_i x_i^2.
\]

Ta có:

\[
\sum_{i=a}^b \mu_i(c-x_i)=Mc-U.
\]

Và:

\[
\sum_{i=a}^b h_i(c-x_i)^2
= Ac^2 - 2Bc + C.
\]

Do đó interval objective là:

\[
F_{a:b}(c)
= \lambda_b(Mc-U)^2 + Ac^2 - 2Bc + C.
\]

Mở rộng:

\[
F_{a:b}(c)
= (\lambda_b M^2 + A)c^2
-2(\lambda_b MU+B)c
+ \lambda_b U^2+C.
\]

Đây là quadratic theo `c`.

Codeword tối ưu là:

\[
\boxed{
c^*_{a:b}
=
\frac{\lambda_b MU+B}{\lambda_b M^2+A}
}
\]

với điều kiện mẫu số khác 0.

Cost tối ưu là:

\[
\boxed{
\operatorname{Cost}(a,b)
=
\lambda_b U^2+C
-
\frac{(\lambda_b MU+B)^2}{\lambda_b M^2+A}
}
\]

Nếu:

\[
\lambda_b M^2 + A \approx 0,
\]

thì interval gần như không có weight/sensitivity/bias information. Trong code nên xử lý bằng epsilon:

```python
denom = bias_lambda * M * M + A
if denom <= eps:
    c = mean(x[a:b+1])
    cost = 0.0
else:
    c = (bias_lambda * M * U + B) / denom
    cost = bias_lambda * U * U + C - (bias_lambda * M * U + B)**2 / denom
```

---

## 5. Prefix sums

Sau khi sort weights, ta cần sort luôn `mu` và `h` theo cùng thứ tự.

Gọi:

```text
x: sorted weights
mu: activation mean sorted according to x
h: diagonal Hessian/Fisher/variance sorted according to x
```

Precompute prefix sums:

\[
P_M[t] = \sum_{i=1}^t \mu_i,
\]

\[
P_U[t] = \sum_{i=1}^t \mu_i x_i,
\]

\[
P_A[t] = \sum_{i=1}^t h_i,
\]

\[
P_B[t] = \sum_{i=1}^t h_i x_i,
\]

\[
P_C[t] = \sum_{i=1}^t h_i x_i^2.
\]

Trong code Python nên dùng 0-based indexing và prefix length `n+1`:

```python
PM[0] = 0
PM[t+1] = PM[t] + mu[t]
```

Tổng trên interval inclusive `[l, r]` là:

```python
M = PM[r+1] - PM[l]
U = PU[r+1] - PU[l]
A = PA[r+1] - PA[l]
B = PB[r+1] - PB[l]
C = PC[r+1] - PC[l]
```

Nhờ vậy `interval_cost(l, r)` tính trong `O(1)`.

---

## 6. Solver 1: BV-HierSplit

### 6.1 Ý tưởng

`BV-HierSplit` là solver phân cấp theo level.

Nó bắt đầu từ một interval chứa toàn bộ row:

\[
[0,n-1].
\]

Mỗi level, nó split tất cả interval hiện tại thành hai interval con.

Nếu target `K` là power-of-two, ví dụ:

```text
K = 2, 4, 8, 16
```

thì quá trình là:

```text
1 interval
→ 2 intervals
→ 4 intervals
→ 8 intervals
→ 16 intervals
```

Cách này giống tinh thần two-split hierarchical quantization.

### 6.2 Best split của một interval

Với interval `[l, r]`, thử mọi split `t`:

```text
[l, t] and [t+1, r]
```

với:

```text
l <= t < r
```

Gain:

\[
\operatorname{Gain}(l,r,t)
=
\operatorname{Cost}(l,r)
-
\operatorname{Cost}(l,t)
-
\operatorname{Cost}(t+1,r).
\]

Chọn:

\[
t^* = \arg\max_t \operatorname{Gain}(l,r,t).
\]

Nếu interval có ít hơn 2 điểm, không split được.

### 6.3 Pseudo-code

```python
def best_split(l, r, cost_fn, min_size=1):
    """
    Find best binary split for interval [l, r].
    Returns: best_gain, best_t
    """
    if r - l + 1 < 2 * min_size:
        return -float("inf"), None

    base = cost_fn(l, r)
    best_gain = -float("inf")
    best_t = None

    # ensure both children have at least min_size elements
    t_min = l + min_size - 1
    t_max = r - min_size

    for t in range(t_min, t_max + 1):
        left = cost_fn(l, t)
        right = cost_fn(t + 1, r)
        gain = base - left - right
        if gain > best_gain:
            best_gain = gain
            best_t = t

    return best_gain, best_t
```

```python
def bv_hier_split(x, mu, h, K, bias_lambda=1.0, min_size=1, eps=1e-12):
    """
    Hierarchical two-split BV scalar quantization.

    Inputs:
        x: 1D sorted weights, shape [n]
        mu: sorted activation means, shape [n]
        h: sorted diagonal Hessian/Fisher/variance, shape [n]
        K: target number of intervals/codewords, should be power of two
        bias_lambda: default 1.0
        min_size: minimum number of points per interval

    Outputs:
        intervals: list of (l, r)
        codewords: list of c values, one per interval
    """
    prefix = build_prefix_sums(x, mu, h)
    cost_fn = lambda l, r: interval_cost(prefix, l, r, bias_lambda, eps)[0]

    intervals = [(0, len(x) - 1)]

    while len(intervals) < K:
        new_intervals = []
        for (l, r) in intervals:
            gain, t = best_split(l, r, cost_fn, min_size=min_size)
            if t is None:
                # cannot split, keep as is
                new_intervals.append((l, r))
            else:
                new_intervals.append((l, t))
                new_intervals.append((t + 1, r))

        intervals = new_intervals

        # If no new interval was created, stop to avoid infinite loop.
        if len(intervals) == len(new_intervals) and len(intervals) >= K:
            break

        # If we overshoot K due to non-power-of-two handling, trim or use greedy version instead.
        if len(intervals) >= K:
            intervals = intervals[:K]
            break

    codewords = []
    for (l, r) in intervals:
        cost, c = interval_cost(prefix, l, r, bias_lambda, eps)
        codewords.append(c)

    return intervals, codewords
```

### 6.4 Ưu điểm

- Rất đơn giản.
- Rất dễ code.
- Tạo codebook phân cấp đẹp.
- Hợp nếu muốn support nhiều bitwidth theo kiểu any-precision.

### 6.5 Nhược điểm

- Mỗi level split tất cả interval, kể cả interval có gain nhỏ.
- Không tối ưu tốt nhất cho một target `K` cố định.
- Nếu distribution không đều, có thể phí codeword cho vùng ít quan trọng.

---

## 7. Solver 2: BV-GreedySplit

### 7.1 Ý tưởng

`BV-GreedySplit` cũng bắt đầu từ một interval chứa toàn bộ row.

Nhưng mỗi bước nó chỉ split **một interval có gain lớn nhất**.

Pipeline:

```text
1 interval
→ split interval tốt nhất
→ split interval tốt nhất tiếp theo
→ ...
→ đủ K intervals
```

Điều này giúp phân bổ codeword linh hoạt hơn.

Nếu một vùng weight có nhiều điểm, sensitivity cao, hoặc bias-variance cost lớn, nó có thể được split nhiều lần hơn.

### 7.2 Pseudo-code dùng priority queue

Mỗi interval trong heap lưu:

```text
-gain, l, r, t
```

Dùng `-gain` vì Python `heapq` là min-heap.

```python
import heapq


def bv_greedy_split(x, mu, h, K, bias_lambda=1.0, min_size=1, eps=1e-12):
    """
    Greedy best-split BV scalar quantization.

    Inputs:
        x: 1D sorted weights, shape [n]
        mu: sorted activation means, shape [n]
        h: sorted diagonal Hessian/Fisher/variance, shape [n]
        K: target number of intervals/codewords
        bias_lambda: default 1.0
        min_size: minimum interval size

    Outputs:
        intervals: list of (l, r)
        codewords: list of c values
    """
    n = len(x)
    prefix = build_prefix_sums(x, mu, h)
    cost_fn = lambda l, r: interval_cost(prefix, l, r, bias_lambda, eps)[0]

    intervals = set()
    intervals.add((0, n - 1))

    heap = []
    gain, t = best_split(0, n - 1, cost_fn, min_size=min_size)
    if t is not None:
        heapq.heappush(heap, (-gain, 0, n - 1, t))

    while len(intervals) < K and len(heap) > 0:
        neg_gain, l, r, t = heapq.heappop(heap)
        gain = -neg_gain

        # Skip stale heap item if interval no longer exists.
        if (l, r) not in intervals:
            continue

        # If gain is non-positive, splitting no longer improves the objective.
        # Depending on implementation, one may still split to reach exactly K.
        # For quantization, we usually still want exactly K codewords.
        # Therefore this check can be disabled.
        # if gain <= 0:
        #     break

        intervals.remove((l, r))
        left_interval = (l, t)
        right_interval = (t + 1, r)
        intervals.add(left_interval)
        intervals.add(right_interval)

        # Compute best split for left child.
        lgain, lt = best_split(left_interval[0], left_interval[1], cost_fn, min_size=min_size)
        if lt is not None:
            heapq.heappush(heap, (-lgain, left_interval[0], left_interval[1], lt))

        # Compute best split for right child.
        rgain, rt = best_split(right_interval[0], right_interval[1], cost_fn, min_size=min_size)
        if rt is not None:
            heapq.heappush(heap, (-rgain, right_interval[0], right_interval[1], rt))

    # Sort intervals by left boundary.
    intervals = sorted(list(intervals), key=lambda p: p[0])

    codewords = []
    for (l, r) in intervals:
        cost, c = interval_cost(prefix, l, r, bias_lambda, eps)
        codewords.append(c)

    return intervals, codewords
```

### 7.3 Ưu điểm

- Vẫn rất nhanh.
- Chất lượng thường tốt hơn hierarchical split cho một target bitwidth cố định.
- Không cần initialization.
- Phân bổ codeword theo gain thực sự của objective.

### 7.4 Nhược điểm

- Không có guarantee global optimal.
- Tree có thể lệch, codebook không đẹp bằng hierarchical nếu muốn nested any-precision.
- Quyết định split sớm có thể ảnh hưởng split sau.

---

## 8. Helper functions cần code

### 8.1 Sort inputs

Input ban đầu thường là unsorted weight row:

```python
w_row: shape [n]
mu: shape [n]
h: shape [n]
```

Sort theo `w_row`:

```python
order = np.argsort(w_row)
x = w_row[order]
mu_sorted = mu[order]
h_sorted = h[order]
```

Nếu dùng PyTorch:

```python
x, order = torch.sort(w_row)
mu_sorted = mu[order]
h_sorted = h[order]
```

Sau khi có intervals/codewords trên sorted order, cần map assignment về original order.

### 8.2 Build prefix sums

```python
def build_prefix_sums(x, mu, h):
    """
    x, mu, h are 1D arrays of length n.
    Returns dict of prefix arrays of length n+1.
    """
    n = len(x)
    PM = zeros(n + 1)
    PU = zeros(n + 1)
    PA = zeros(n + 1)
    PB = zeros(n + 1)
    PC = zeros(n + 1)

    for i in range(n):
        PM[i + 1] = PM[i] + mu[i]
        PU[i + 1] = PU[i] + mu[i] * x[i]
        PA[i + 1] = PA[i] + h[i]
        PB[i + 1] = PB[i] + h[i] * x[i]
        PC[i + 1] = PC[i] + h[i] * x[i] * x[i]

    return {"PM": PM, "PU": PU, "PA": PA, "PB": PB, "PC": PC}
```

### 8.3 Interval stats

```python
def interval_stats(prefix, l, r):
    PM, PU, PA, PB, PC = prefix["PM"], prefix["PU"], prefix["PA"], prefix["PB"], prefix["PC"]

    M = PM[r + 1] - PM[l]
    U = PU[r + 1] - PU[l]
    A = PA[r + 1] - PA[l]
    B = PB[r + 1] - PB[l]
    C = PC[r + 1] - PC[l]

    return M, U, A, B, C
```

### 8.4 Interval cost and codeword

```python
def interval_cost(prefix, l, r, bias_lambda=1.0, eps=1e-12):
    M, U, A, B, C = interval_stats(prefix, l, r)

    denom = bias_lambda * M * M + A
    numer = bias_lambda * M * U + B

    if denom <= eps:
        # Fallback. In practice this case is rare if h is positive.
        # Return zero cost and any stable representative.
        c = 0.0
        cost = 0.0
    else:
        c = numer / denom
        cost = bias_lambda * U * U + C - (numer * numer) / denom

        # Numerical guard: cost should be nonnegative up to floating error.
        if cost < 0 and cost > -1e-8:
            cost = 0.0

    return cost, c
```

Better fallback if access to `x` is available:

```python
c = x[l:r+1].mean()
```

---

## 9. Build final assignment

After intervals and codewords are obtained in sorted order, build assignment labels.

```python
def build_assignment(n, intervals, order, codewords):
    """
    n: number of weights
    intervals: sorted-order intervals [(l,r), ...]
    order: argsort order such that x = w[order]
    codewords: list of codewords

    Returns:
        labels_original: label for each original index
        q_original: quantized values in original order
    """
    labels_sorted = np.empty(n, dtype=np.int64)
    q_sorted = np.empty(n, dtype=np.float32)

    for k, (l, r) in enumerate(intervals):
        labels_sorted[l:r+1] = k
        q_sorted[l:r+1] = codewords[k]

    labels_original = np.empty(n, dtype=np.int64)
    q_original = np.empty(n, dtype=np.float32)

    labels_original[order] = labels_sorted
    q_original[order] = q_sorted

    return labels_original, q_original
```

For PyTorch, use tensor indexing similarly.

---

## 10. Complexity

Let:

- `n` = number of weights in a row/output channel.
- `K` = number of codewords.

Sorting:

\[
O(n\log n).
\]

Prefix sums:

\[
O(n).
\]

A single best split over one interval of length `m` costs:

\[
O(m).
\]

For `BV-HierSplit`, total cost is roughly:

\[
O(n\log K)
\]

if every level scans disjoint intervals whose total length is `n`.

For `BV-GreedySplit`, worst-case cost can be:

\[
O(Kn)
\]

because some large intervals may be scanned multiple times. Since `K` is small, e.g. 8 or 16, this is still practical.

In practice:

```text
K = 8  for 3-bit
K = 16 for 4-bit
n ≈ 4096 or 8192
```

so `O(Kn)` per row is acceptable compared with exact DP `O(Kn^2)`.

---

## 11. Recommended default choices

Recommended defaults:

```python
bias_lambda = 1.0
min_size = 1
eps = 1e-12
solver = "greedy"
```

Use `BV-GreedySplit` as the main solver if target is best quality at a fixed bitwidth.

Use `BV-HierSplit` if nested/any-precision codebooks are important.

---

## 12. Comparison between the two solvers

| Solver | Main idea | Speed | Quality for fixed K | Nested codebook | Recommended use |
|---|---|---:|---:|---:|---|
| `BV-HierSplit` | split every interval at each level | very fast | good | very good | any-precision / simple baseline |
| `BV-GreedySplit` | split interval with largest gain | fast | usually better | less regular | main fixed-bit solver |

Short recommendation:

```text
If only one target bitwidth matters: use BV-GreedySplit.
If many bitwidths must share structure: use BV-HierSplit.
```

---

## 13. Relation to RBVT

RBVT optimizes final assignment on a fixed codebook. It uses a bias-variance view of layer output error:

\[
(\mu^\top e)^2 + e^\top \Sigma e.
\]

BV-SQ moves this idea earlier into codebook construction.

Difference:

```text
RBVT:
    fixed codebook
    optimize assignment moves

BV-SQ:
    construct codebook from scratch
    optimize sorted interval boundaries and codeword values
```

The proposed interval objective is not the full dense covariance loss. It is a diagonal/local-bias surrogate designed to preserve the 1D sorted interval structure.

---

## 14. Relation to SqueezeLLM / sqllm

SqueezeLLM-style scalar quantization uses a weighted reconstruction cost:

\[
\sum_i s_i(w_i-c)^2.
\]

BV-SQ replaces this interval cost by:

\[
\lambda_b
\left(\sum_i \mu_i(c-w_i)\right)^2
+
\sum_i h_i(c-w_i)^2.
\]

Thus:

```text
SqueezeLLM/sqllm:
    sensitivity-weighted scalar reconstruction

BV-SQ:
    diagonal variance reconstruction
    + local first-moment bias correction
```

Both methods preserve:

```text
sort weights → interval boundaries → codewords
```

---

## 15. Suggested experiments / ablations

Recommended variants:

1. `sqllm` baseline.
2. `BV-HierSplit`, `bias_lambda=1.0`.
3. `BV-GreedySplit`, `bias_lambda=1.0`.
4. `BV-GreedySplit` with `bias_lambda=0.0`.
   - This reduces to diagonal variance-only interval quantization.
5. `BV-GreedySplit + RBVT final assignment`.
6. Optional: exact DP on a small subset of rows/layers to estimate how close greedy is to optimal under the proposed interval cost.

Important ablation:

```text
bias_lambda = 0 vs bias_lambda = 1
```

This tests whether adding local mean-shift penalty helps beyond diagonal variance weighting.

---

## 16. Paper-style method wording

A concise method paragraph:

> We propose Bias-Variance Scalar Quantization (BV-SQ), a from-scratch non-uniform codebook construction method for weight-only LLM quantization. Like SqueezeLLM-style scalar quantization, BV-SQ sorts each weight row and represents codebook assignments as contiguous 1D intervals. However, instead of minimizing only a sensitivity-weighted reconstruction objective, BV-SQ uses a bias-variance interval surrogate derived from the layer output error decomposition. For each interval, the optimal codeword has a closed-form solution that combines a local first-moment shift term and a diagonal covariance-weighted variance term. This allows constant-time interval cost evaluation using prefix sums. We construct the codebook using either hierarchical two-split quantization or greedy best-gain splitting, both of which preserve the sorted interval structure and avoid expensive dynamic programming.

---

## 17. Minimal implementation checklist for Codex

Implement these functions:

```python
sort_row(w_row, mu, h)
build_prefix_sums(x, mu_sorted, h_sorted)
interval_stats(prefix, l, r)
interval_cost(prefix, l, r, bias_lambda=1.0, eps=1e-12)
best_split(l, r, cost_fn, min_size=1)
bv_hier_split(x, mu_sorted, h_sorted, K, bias_lambda=1.0)
bv_greedy_split(x, mu_sorted, h_sorted, K, bias_lambda=1.0)
build_assignment(n, intervals, order, codewords)
```

Expected output per row:

```python
codewords: shape [K]
labels: shape [n]
q_row: shape [n]
```

For a whole weight matrix `W` of shape `[out_features, in_features]`, run per row:

```python
for j in range(out_features):
    w_row = W[j, :]
    labels_j, codewords_j, q_row_j = bv_sq_row(w_row, mu, h, K)
```

Here `mu` and `h` are activation statistics over input dimensions, shared across output rows of the same linear layer.

---

## 18. Final summary

BV-SQ is a simple extension of sorted scalar non-uniform quantization:

```text
SqueezeLLM-style:
    interval cost = sensitivity-weighted MSE

BV-SQ:
    interval cost = diagonal variance MSE + local mean-shift bias penalty
```

The key formulas are:

\[
c^*_{a:b}
=
\frac{\lambda_b MU+B}{\lambda_b M^2+A}
\]

and

\[
\operatorname{Cost}(a,b)
=
\lambda_b U^2+C
-
\frac{(\lambda_b MU+B)^2}{\lambda_b M^2+A}.
\]

with default:

\[
\lambda_b=1.
\]

The recommended solver is:

```text
BV-GreedySplit for best fixed-bit quality.
BV-HierSplit for simple nested multi-bit codebooks.
```

