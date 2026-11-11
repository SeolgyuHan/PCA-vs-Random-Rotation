// Fused kernels for the streaming big-set experiments (G866). Build: make (g++ -O3 -march=native -fopenmp).
#include <cstdint>
#include <cmath>
#include <vector>
#include <omp.h>
extern "C" {
// out[i,j] = clamp(floor((X[i,j]-mn[j])*iw[j]), 0, Lm1[j]); components with Lm1[j] < 0 (no bits / degenerate) -> 0.
void quant_idx(const float* X, int64_t n, int d, const float* mn, const float* iw, const float* Lm1, float* out) {
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
    const float* x = X + i * d; float* o = out + i * d;
    for (int j = 0; j < d; j++) {
      float v = std::floor((x[j] - mn[j]) * iw[j]);
      v = v < Lm1[j] ? v : Lm1[j]; v = v < 0.f ? 0.f : v;
      o[j] = Lm1[j] < 0.f ? 0.f : v;
    }
  }
}
// entries S[q, j] > theta[q]  (S is nq x n, row-major). Returns the count; fills up to cap entries (q, j, s).
// If the count exceeds cap, returns -count and fills nothing.
int64_t filter_gt(const float* S, int nq, int64_t n, const float* theta, int32_t* oq, int32_t* oj, float* os, int64_t cap) {
  std::vector<int64_t> cnt(nq + 1, 0);
  #pragma omp parallel for schedule(dynamic, 4)
  for (int q = 0; q < nq; q++) { const float* s = S + (int64_t)q * n; const float t = theta[q]; int64_t c = 0; for (int64_t j = 0; j < n; j++) c += (s[j] > t); cnt[q + 1] = c; }
  for (int q = 0; q < nq; q++) cnt[q + 1] += cnt[q];
  if (cnt[nq] > cap) return -cnt[nq];
  #pragma omp parallel for schedule(dynamic, 4)
  for (int q = 0; q < nq; q++) { const float* s = S + (int64_t)q * n; const float t = theta[q]; int64_t o = cnt[q];
    for (int64_t j = 0; j < n; j++) if (s[j] > t) { oq[o] = q; oj[o] = (int32_t)j; os[o] = s[j]; o++; } }
  return cnt[nq];
}
// IVF probe levels: l = lvl_of_rank[rank[q, lst[j]]] (255 = beyond the deepest level); emit if l < L and S[q,j] > theta[q*L + l].
int64_t filter_lvl(const float* S, int nq, int64_t n, const int32_t* lst, const int16_t* rank, int nlist, const uint8_t* lvl_of_rank,
                   const float* theta, int L, int32_t* oq, int32_t* oj, uint8_t* ol, float* os, int64_t cap) {
  std::vector<int64_t> cnt(nq + 1, 0);
  #pragma omp parallel for schedule(dynamic, 4)
  for (int q = 0; q < nq; q++) { const float* s = S + (int64_t)q * n; const int16_t* rk = rank + (int64_t)q * nlist; const float* th = theta + (int64_t)q * L; int64_t c = 0;
    for (int64_t j = 0; j < n; j++) { uint8_t l = lvl_of_rank[rk[lst[j]]]; c += (l < L) && (s[j] > th[l]); } cnt[q + 1] = c; }
  for (int q = 0; q < nq; q++) cnt[q + 1] += cnt[q];
  if (cnt[nq] > cap) return -cnt[nq];
  #pragma omp parallel for schedule(dynamic, 4)
  for (int q = 0; q < nq; q++) { const float* s = S + (int64_t)q * n; const int16_t* rk = rank + (int64_t)q * nlist; const float* th = theta + (int64_t)q * L; int64_t o = cnt[q];
    for (int64_t j = 0; j < n; j++) { uint8_t l = lvl_of_rank[rk[lst[j]]]; if ((l < L) && (s[j] > th[l])) { oq[o] = q; oj[o] = (int32_t)j; ol[o] = l; os[o] = s[j]; o++; } } }
  return cnt[nq];
}
// column min / max of X (n x d) accumulated into mn / mx (initialised by the caller)
void minmax_cols(const float* X, int64_t n, int d, float* mn, float* mx) {
  int nt = omp_get_max_threads(); std::vector<float> lmn((size_t)nt * d, 3.4e38f), lmx((size_t)nt * d, -3.4e38f);
  #pragma omp parallel
  { int t = omp_get_thread_num(); float* a = lmn.data() + (size_t)t * d; float* b = lmx.data() + (size_t)t * d;
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; i++) { const float* x = X + i * d; for (int j = 0; j < d; j++) { a[j] = x[j] < a[j] ? x[j] : a[j]; b[j] = x[j] > b[j] ? x[j] : b[j]; } } }
  for (int t = 0; t < nt; t++) for (int j = 0; j < d; j++) { mn[j] = lmn[(size_t)t * d + j] < mn[j] ? lmn[(size_t)t * d + j] : mn[j]; mx[j] = lmx[(size_t)t * d + j] > mx[j] ? lmx[(size_t)t * d + j] : mx[j]; }
}
// dot products of selected pairs: out[k] = <A[ia[k]], B[ib[k]]>, A (na x d), B (nb x d)
void pair_dots(const float* A, const float* B, int d, const int64_t* ia, const int64_t* ib, int64_t npairs, float* out) {
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < npairs; k++) { const float* a = A + ia[k] * d; const float* b = B + ib[k] * d; float s = 0.f; for (int j = 0; j < d; j++) s += a[j] * b[j]; out[k] = s; }
}

// fused scoring + filtering for low dimension: for every row r (block-parallel) and query q, s = <Q[q], X[r]>; emit if s > theta[q].
// No nq x n score matrix is materialised (memory traffic of the separate sgemm + filter path removed). X (n x d), Q (nq x d), row-major.
static inline float dot_f(const float* __restrict a, const float* __restrict b, int d) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; int k = 0;
  for (; k + 8 <= d; k += 8) { for (int l = 0; l < 8; l++) acc[l] += a[k + l] * b[k + l]; }
  float s = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
  for (; k < d; k++) s += a[k] * b[k];
  return s;
}
int64_t score_filter(const float* X, int64_t n, int d, const float* Q, int nq, const float* theta, int32_t* oq, int32_t* oj, float* os, int64_t cap) {
  int nt = omp_get_max_threads(); std::vector<std::vector<int32_t>> lq(nt), lj(nt); std::vector<std::vector<float>> ls(nt);
  #pragma omp parallel
  { int t = omp_get_thread_num(); std::vector<int32_t>& vq = lq[t]; std::vector<int32_t>& vj = lj[t]; std::vector<float>& vs = ls[t];
    #pragma omp for schedule(dynamic, 256)
    for (int64_t r = 0; r < n; r++) { const float* x = X + r * d;
      for (int q = 0; q < nq; q++) { float s = dot_f(Q + (int64_t)q * d, x, d); if (s > theta[q]) { vq.push_back(q); vj.push_back((int32_t)r); vs.push_back(s); } } } }
  int64_t total = 0; for (int t = 0; t < nt; t++) total += (int64_t)lq[t].size();
  if (total > cap) return -total;
  int64_t o = 0; for (int t = 0; t < nt; t++) { size_t m = lq[t].size(); if (m) { std::copy(lq[t].begin(), lq[t].end(), oq + o); std::copy(lj[t].begin(), lj[t].end(), oj + o); std::copy(ls[t].begin(), ls[t].end(), os + o); o += (int64_t)m; } }
  return total;
}
}
