#!/usr/bin/env python3
"""
Q10 Attention — tehnika: Quantum Self-Attention Network (QSAN) preko SWAP-test-a
(čisto kvantno, bez klasičnog softmax-a i bez hibrida).

Arhitektura (Self-Attention, key = value):
  - Query stanje |Q⟩: amplitude-encoding freq_vector-a CELOG CSV-a (dim 2^nq).
  - Blokovi CSV-a (B blokova, redovi se dele uzastopno): za svaki blok b
    amplitude-encoding freq_vector-a tog bloka → |K_b⟩ = |V_b⟩.
  - Attention težina (kvantno, bez softmax-a): SWAP-test kolo nad
    (ancilla, Q-registar, K_b-registar) → P(ancilla = 0) = (1 + |⟨Q|K_b⟩|²)/2
    → w_b = |⟨Q|K_b⟩|² = 2·P(0) − 1, ekstrakcija iz egzaktnog Statevector-a.
  - Spoj vrednosti: p_out = Σ_b w_b · |K_b|²  (normalizovano), pa bias_39 → NEXT.

Sve deterministički: seed=39; amp_Q i amp_K iz CELOG CSV-a.
Deterministička grid-optimizacija (nq, B) po meri cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_B = (4, 5, 7, 8)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


# =========================
# Amplitude-encoding (dim = 2^nq) iz freq-vektora
# =========================
def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


def block_freqs(H: np.ndarray, B: int) -> List[np.ndarray]:
    """CSV se deli u B uzastopnih blokova po redovima; vraća freq_vector po bloku."""
    n = H.shape[0]
    edges = np.linspace(0, n, B + 1, dtype=int)
    out: List[np.ndarray] = []
    for i in range(B):
        if edges[i + 1] <= edges[i]:
            out.append(np.zeros(N_MAX, dtype=np.float64))
        else:
            out.append(freq_vector(H[edges[i] : edges[i + 1]]))
    return out


# =========================
# SWAP test — kvantno |⟨Q|K⟩|² iz egzaktnog Statevector-a
# =========================
def swap_test_overlap_sq(nq: int, amp_q: np.ndarray, amp_k: np.ndarray) -> float:
    """
    Kolo: ancilla (qubit 0), zatim Q-registar (nq), pa K-registar (nq).
    P(ancilla = 0) = (1 + |⟨Q|K⟩|²) / 2  →  |⟨Q|K⟩|² = 2·P(0) − 1.
    """
    total = 1 + 2 * nq
    qc = QuantumCircuit(total, name="swap_test")

    qc.append(StatePreparation(amp_q.tolist()), list(range(1, 1 + nq)))
    qc.append(StatePreparation(amp_k.tolist()), list(range(1 + nq, 1 + 2 * nq)))

    qc.h(0)
    for i in range(nq):
        qc.cswap(0, 1 + i, 1 + nq + i)
    qc.h(0)

    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2
    dim = 2 ** total
    # Qiskit little-endian: qubit 0 = najniži bit
    p_anc0 = float(sum(p[i] for i in range(dim) if (i & 1) == 0))
    w = 2.0 * p_anc0 - 1.0
    return float(max(0.0, w))


# =========================
# QSAN — Self-Attention agregacija
# =========================
def qsan_state_probs(H: np.ndarray, nq: int, B: int) -> np.ndarray:
    """Vraća distribuciju nad 2^nq stanja posle attention-weighted spoja blokova."""
    f_csv = freq_vector(H)
    amp_q = amp_from_freq(f_csv, nq)

    f_blocks = block_freqs(H, B)
    amps_k: List[np.ndarray] = [amp_from_freq(fb, nq) for fb in f_blocks]

    weights = np.array([swap_test_overlap_sq(nq, amp_q, ak) for ak in amps_k], dtype=np.float64)
    s_w = float(weights.sum())
    if s_w < 1e-18:
        weights = np.ones(B, dtype=np.float64) / B
    else:
        weights = weights / s_w

    dim = 2 ** nq
    out = np.zeros(dim, dtype=np.float64)
    for w, ak in zip(weights, amps_k):
        out += float(w) * (ak ** 2)
    s = float(out.sum())
    return out / s if s > 0 else out


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, B) po meri cos(bias_39, freq_csv)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for B in GRID_B:
            try:
                p = qsan_state_probs(H, nq, B)
                b = bias_39(p)
                score = cosine(b, f_csv_n)
            except Exception:
                continue
            key = (score, -nq, -B)
            if best is None or key > best[0]:
                best = (key, dict(nq=nq, B=B, score=float(score)))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q10 Attention (QSAN — SWAP-test): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| B (blokova):", best["B"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    p = qsan_state_probs(H, best["nq"], best["B"])
    pred = pick_next_combination(p)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q10 Attention (QSAN — SWAP-test): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 5 | B (blokova): 8 | cos(bias, freq_csv): 0.900349
predikcija NEXT: (7, 19, 22, 24, 27, 28, 31)
"""



"""
Q10_Attention_QSAN.py — tehnika: Quantum Self-Attention Network (SWAP-test).

Query = amplitude-encoding ukupnog freq_vector-a (CEO CSV).
Key/Value = amplitude-encoding freq_vector-a po bloku (CEO CSV → B uzastopnih blokova).
Attention težine = |⟨Q|K_b⟩|² iz SWAP-test kola (P(anc=0) na egzaktnom Statevector-u,
bez klasičnog softmax-a).
Output p = Σ_b w_b · |K_b|², p → bias_39 → TOP-7 = NEXT.

Tehnike:
SWAP-test za kvantnu meru sličnosti stanja.
Amplitude encoding (StatePreparation) za Q i K.
Egzaktni Statevector simulator (bez uzorkovanja).
Deterministička grid-optimizacija (nq, B).

Prednosti:
Čisto kvantno — attention težine proizlaze iz interferencije (SWAP test),
ne iz klasičnog softmax-a. 
Prirodno proširivo na višeglavnu pažnju (više query-ja).

Nedostaci:
Per blok koristi 2·nq + 1 qubit-a za SWAP test (qc budžet je okej pri nq ≤ 6).
Aggregacija blokova je marginala preko w_b (linearna), nije LCU-unitary.
Mera cos(bias, freq_csv) je pristrana ka reprodukciji marginale.
mod-39 readout meša stanja (dim 2^nq ≠ 39).
"""
