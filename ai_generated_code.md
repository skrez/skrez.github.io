```
###############################################################################
#   Quaternion convexity search (n = 1)  •  Sage 10.x / Python 3.x
#
#   Looks for a quadratic ρ(q) = ½ qᵀ H q whose Hessian H fails H-convexity
#   but passes every sampled (I_v, J_v, K_v)-convexity test.
#
#   All code is fully annotated — feel free to adapt / extend to n > 1.
###############################################################################

from sage.all import *
import random, math

# ---------------------------------------------------------------------------
# 1)  Quaternion arithmetic helpers
# ---------------------------------------------------------------------------

def q_mult(p, q):
    """Hamilton product of quaternions p, q given as 4-tuples (w,x,y,z)."""
    w1,x1,y1,z1 = p
    w2,x2,y2,z2 = q
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)

def q_conj(q):
    """Quaternion conjugate."""
    w,x,y,z = q
    return (w, -x, -y, -z)

def q_inv(q):
    """
    Multiplicative inverse of a non-zero quaternion.
    Works for ANY norm; raises ZeroDivisionError on q = 0.
    """
    w,x,y,z = q
    norm_sq = w*w + x*x + y*y + z*z
    if norm_sq == 0:
        raise ZeroDivisionError("Quaternion 0 has no inverse.")
    return (w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq)

def q_conjugate(v, p):
    """Return v p v^{-1} (quaternion conjugation)."""
    return q_mult(v, q_mult(p, q_inv(v)))

def random_unit_quaternion():
    """Pick a random point on the 3-sphere S³ via normalised Gaussians."""
    v = vector(RR, [random.gauss(0,1) for _ in range(4)])
    return tuple(v / v.norm())

# Imaginary-unit basis quaternions
i_quat = (0,1,0,0)
j_quat = (0,0,1,0)
k_quat = (0,0,0,1)

# ---------------------------------------------------------------------------
# 2)  4 × 4 real left-multiplication matrices  I, J, K  on ℍ ≅ ℝ⁴
# ---------------------------------------------------------------------------

def left_mult_blocks():
    """Return constant 4×4 matrices for left mult. by i, j, k."""
    I =  matrix([[ 0,-1, 0, 0],
                 [ 1, 0, 0, 0],
                 [ 0, 0, 0,-1],
                 [ 0, 0, 1, 0]])
    J =  matrix([[ 0, 0,-1, 0],
                 [ 0, 0, 0, 1],
                 [ 1, 0, 0, 0],
                 [ 0,-1, 0, 0]])
    K =  matrix([[ 0, 0, 0,-1],
                 [ 0, 0,-1, 0],
                 [ 0, 1, 0, 0],
                 [ 1, 0, 0, 0]])
    return I, J, K

I_mat, J_mat, K_mat = left_mult_blocks()

def left_mult_matrix_of_unit_imag(a, b, c):
    """Matrix for left multiplication by (a i + b j + c k), with a²+b²+c² = 1."""
    return a*I_mat + b*J_mat + c*K_mat

# ---------------------------------------------------------------------------
# 3)  Levi 2-form evaluator  σ_M(v,w)   **FIXED**
# ---------------------------------------------------------------------------

def _col_to_vector(col):
    """Convert a 4×1 (column-matrix) into a length-4 Sage vector."""
    return vector(RR, [col[i, 0] for i in range(col.nrows())])

def sigma_eval(H, M, v, w):
    r"""
    Evaluate

        σ_M(v,w) = −[ (M w)ᵀ H v  − (M v)ᵀ H w ]

    for a constant symmetric Hessian *H*, an imaginary-unit matrix *M*,
    and column-matrices *v*, *w* (all 4 × 1).
    """
    Mw = _col_to_vector(M * w)
    Mv = _col_to_vector(M * v)
    Hv = _col_to_vector(H * v)
    Hw = _col_to_vector(H * w)

    return - ( Mw.dot_product(Hv) - Mv.dot_product(Hw) )


# ---------------------------------------------------------------------------
# 4)  Random admissible triples for the linear constraints
# ---------------------------------------------------------------------------

def random_IJK_triple():
    """
    Return v1, v2, v3 such that  I v1 + J v2 + K v3 = 0.
    Takes v1, v2 freely and solves for v3.
    """
    v1 = vector(RR, [random.gauss(0,1) for _ in range(4)]).column()
    v2 = vector(RR, [random.gauss(0,1) for _ in range(4)]).column()
    v3 = K_mat * (I_mat*v1 + J_mat*v2)
    return v1, v2, v3

def random_IvJvKv_triple(Iv, Jv, Kv):
    """Same generator for a rotated triple (I_v, J_v, K_v)."""
    v1 = vector(RR, [random.gauss(0,1) for _ in range(4)]).column()
    v2 = vector(RR, [random.gauss(0,1) for _ in range(4)]).column()
    v3 = Kv * (Iv*v1 + Jv*v2)
    return v1, v2, v3

# ---------------------------------------------------------------------------
# 5)  (I_v, J_v, K_v)-convexity tester
# ---------------------------------------------------------------------------

def is_rotated_convex(H, v,
                      nsamples = 50,
                      eps      = 1e-10):
    """
    Heuristically test (I_v, J_v, K_v)-convexity of quadratic ρ with Hessian H.
    Returns False immediately on the first violation.
    """
    # Build rotated matrices I_v, J_v, K_v
    i_v = q_conjugate(v, i_quat)
    j_v = q_conjugate(v, j_quat)
    k_v = q_conjugate(v, k_quat)
    Iv = left_mult_matrix_of_unit_imag(i_v[1], i_v[2], i_v[3])
    Jv = left_mult_matrix_of_unit_imag(j_v[1], j_v[2], j_v[3])
    Kv = left_mult_matrix_of_unit_imag(k_v[1], k_v[2], k_v[3])

    for _ in range(nsamples):
        v1, v2, v3 = random_IvJvKv_triple(Iv, Jv, Kv)
        c = ( sigma_eval(H, Iv, v2, v3)
            + sigma_eval(H, Jv, v3, v1)
            + sigma_eval(H, Kv, v1, v2) )
        if c >= -eps:
            return False          # violates the strict inequality
    return True

# ---------------------------------------------------------------------------
# 6)  FULL H-convexity tester (corrected)
# ---------------------------------------------------------------------------

def is_H_convex_full(H,
                     nsamples = 400,
                     eps      = 1e-10):
    """
    Monte-Carlo test of H-convexity for a quadratic ρ with Hessian H.

    For each sample:
      – choose A1, A2, A3 freely (Gaussian),
      – set A0 = I A1 + J A2 + K A3   to satisfy  A0 − I A1 − J A2 − K A3 = 0,
      – compute the coefficient  c(A)  from formula (★).
    """
    for _ in range(nsamples):
        A1 = vector(RR, [random.gauss(0,1) for _ in range(4)]).column()
        A2 = vector(RR, [random.gauss(0,1) for _ in range(4)]).column()
        A3 = vector(RR, [random.gauss(0,1) for _ in range(4)]).column()
        A0 = I_mat*A1 + J_mat*A2 + K_mat*A3            # ensures constraint

        c = ( sigma_eval(H, I_mat, A2, A3)
            + sigma_eval(H, I_mat, A0, A1)
            - sigma_eval(H, J_mat, A1, A3)
            + sigma_eval(H, J_mat, A0, A2)
            + sigma_eval(H, K_mat, A1, A2)
            + sigma_eval(H, K_mat, A0, A3) )

        if c >= -eps:              # violation detected
            return False
    return True                    # all samples strictly negative

# ---------------------------------------------------------------------------
# 7)  Random symmetric 4 × 4 Hessian generator
# ---------------------------------------------------------------------------

def random_symmetric_matrix(range_int = (-3, 3)):
    """Random integer symmetric 4×4 matrix in the given range."""
    entries = [[ZZ(random.randint(*range_int)) for _ in range(4)] for _ in range(4)]
    H = matrix(4,4, entries)
    return (H + H.transpose())/2

# ---------------------------------------------------------------------------
# 8)  Counter-example search driver
# ---------------------------------------------------------------------------

def search_counterexample(max_trials        = 10000,
                          n_v_samples       = 20,
                          n_triples_per_v   = 40,
                          n_H_triples       = 400,
                          int_range         = (-3,3),
                          eps               = 1e-8):
    """
    Monte-Carlo hunt for a quadratic that is (I_v,J_v,K_v)-convex
    for many v but violates H-convexity.

    Prints and returns the first candidate found (or None).
    """
    for trial in range(1, max_trials+1):
        H = random_symmetric_matrix(int_range)

        # Step A – rotated convexity for many v
        all_v_ok = True
        for _ in range(n_v_samples):
            v = random_unit_quaternion()
            if not is_rotated_convex(H, v,
                                     nsamples = n_triples_per_v,
                                     eps      = eps):
                all_v_ok = False
                break
        if not all_v_ok:
            continue                 # fail fast

        # Step B – full H-convexity
        if not is_H_convex_full(H,
                                nsamples = n_H_triples,
                                eps      = eps):
            print("\n=== Candidate counter-example found (trial %d) ===" % trial)
            print("Hessian matrix H =")
            show(H)
            return H

        # progress indicator
        if trial % 500 == 0:
            print("…%d trials, none found yet" % trial)
    print("No counter-example found in %d trials." % max_trials)
    return None

# ---------------------------------------------------------------------------
# 9)  Run a quick search when executed as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Adjust parameters to search deeper / faster.
    search_counterexample(max_trials      = 3000,
                          n_v_samples     = 25,
                          n_triples_per_v = 35,
                          n_H_triples     = 300,
                          int_range       = (-4,4),
                          eps             = 1e-8)
```
