"""Section 4.4, Fig 1c, Appx. Figs 16/17: IP is not activation-difference PPS.

IP is the paper's 3-prompt bank (``runs.IP_BANK``). Vectors:
``steering_vectors/evil_3samp_actdiff_heavy_v1.pt`` (main) and
``steering_vectors/evil_p1_actdiff_heavy_v1.pt`` (single prompt p1,
comparison), built by ``scripts/build_actdiff_vectors.py`` (mean activation
with the prompt prepended minus without, over tokens after the prompt, per
layer; for the bank, averaged over its prompts per sequence; on 2,340 of the
4,681 heavy sequences; metadata in the ``.meta.json`` beside each).

- ``EXTRA_RUNS``: PPS with those vectors at all 28 layers, coefficient 1.0
  (paper Section 4.4), on the heavy data under the standard protocol.
- ``gradient_actdiff`` (GPU): on the first 200 held-out sequences, the
  tokenwise cosine between the IP gradient and the gradient under act-diff
  PPS at coefficients -3..3 (coefficient 0 = no defense, the Section
  4.4 figure's baseline), at every layer, over tokens after the prompt only. IP (3
  prompts): the cosine is taken against the gradient under EACH bank prompt
  (at that prompt's post-prompt tokens), with the bank act-diff vector
  steering, token-pooled per prompt and averaged over the prompts; p1: the
  p1 gradient vs the p1 act-diff vector. Ported from
  scripts/compare_grad_steered_vs_prompted.py.
- Products: the Section 4.4 figure (``fig_ipnotpps_grad``; Fig 1c before
  2026-10-07), Fig 16 (``fig_ip_vs_actdiff``),
  Fig 17 (``fig_actdiff_vec_cos``), and the Section 4.4 behavioral
  comparison (``sec_actdiff_behavior``). Keys without ``p1`` are the main
  (3-prompt) arm; ``*.p1.*`` keys are the single-prompt comparison.
"""

import json
import os
from typing import Dict, List

import numpy as np

from ..macros import Result
from ..runs import ALL_LAYERS, DATA, EVIL
from ..spec import REPO_ROOT, RunSpec
from . import gradient_core as gc
from .. import paper_style as ps
from . import ip_bank
from .base import Analysis, Context, Product

ACTDIFF = "steering_vectors/evil_3samp_actdiff_heavy_v1.pt"  # IP (3 prompts)
ACTDIFF_P1 = "steering_vectors/evil_p1_actdiff_heavy_v1.pt"  # IP p1
ACTDIFF_COEF = 1.0
RUN = "gradient.actdiff_pps"
RUN_P1 = "gradient.actdiff_pps.p1"
IP_RUN, IP_P1_RUN = "grid.base.heavy.ip", "ipcount.1"
NAME = "gradient_actdiff"
N_SEQ = 200
# Appx. grad_ip_is_not_pps.tex:13 (0 is the undefended baseline).
COEFS = (-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5,
         2.0, 3.0)
SHOW_LAYERS = (4, 8, 12, 16, 20, 24, 28)
VARIANTS = {"bank": ACTDIFF, "p1": ACTDIFF_P1}


def _actdiff_run(name: str, vector: str) -> RunSpec:
    return RunSpec(name, DATA["heavy"], "pps", pps_vector=vector,
                   pps_coef=ACTDIFF_COEF, pps_layers=ALL_LAYERS)


EXTRA_RUNS = [_actdiff_run(RUN, ACTDIFF), _actdiff_run(RUN_P1, ACTDIFF_P1)]


def actdiff_meta(path: str = ACTDIFF) -> dict:
    with open(os.path.join(REPO_ROOT, path[:-3] + ".meta.json")) as f:
        return json.load(f)


def heldout_indices() -> List[int]:
    """Held-out sequence indices, in the build script's permutation (the
    bank and p1 vectors must share the split)."""
    splits = [actdiff_meta(p)["split"] for p in VARIANTS.values()]
    if any(sp != splits[0] for sp in splits):
        raise ValueError("act-diff vectors were built on different splits")
    split = splits[0]
    _, held = gc.split_indices(split["n_total"], split["n_build"],
                               split["seed"])
    return held


def coef_cond(coef: float, vector: str = ACTDIFF) -> gc.Condition:
    if coef == 0:
        return gc.NONE
    tag = "" if vector == ACTDIFF else "p1_"
    return gc.Condition(f"actdiff_{tag}{coef:g}", pps_vector=vector,
                        pps_coef=coef, pps_layers=ALL_LAYERS)


def token_pooled(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(coef, layer) table: per member, the mean over all held-out tokens of
    the paired cosine; then the mean over members (equal weight per prompt).

    ``sums``: [member, coef, seq, layer]; ``counts``: [member, seq].
    """
    per_member = sums.sum(axis=2) / counts.sum(axis=1)[:, None, None]
    return per_member.mean(axis=0)


class GradientActdiff(Analysis):
    """cos(grad under IP, grad under act-diff PPS) per layer and coef, for
    IP (3 prompts; bank vector) and IP p1 (p1 vector)."""

    name = NAME
    runs: List[str] = []
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        import torch

        held = heldout_indices()[:N_SEQ]
        chats = gc.load_chats(indices=held)
        probe = gc.GradientProbe(layers=gc.LAYERS)
        ips = gc.ip_bank_conditions()
        p1 = ip_bank.p1_index()
        L, M, K, N = len(gc.LAYERS), len(ips), len(COEFS), len(chats)
        # [variant][member slot, coef, seq, layer]; p1 has one slot.
        slots = {"bank": list(range(M)), "p1": [p1]}
        sums = {v: np.zeros((len(s), K, N, L)) for v, s in slots.items()}
        means = {v: np.zeros_like(a) for v, a in sums.items()}
        counts = {v: np.zeros((len(s), N), dtype=np.int64)
                  for v, s in slots.items()}
        for i, chat in enumerate(chats):
            pp = [probe.run(chat, c, acts=False) for c in ips]
            for variant, vec in VARIANTS.items():
                for k, coef in enumerate(COEFS):
                    ps = probe.run(chat, coef_cond(coef, vec), acts=False)
                    for slot, m in enumerate(slots[variant]):
                        bi, di = gc.aligned_positions(ps.tokens, pp[m].tokens,
                                                      "suffix")
                        if len(bi) == 0:
                            raise RuntimeError(f"held-out seq {held[i]}: "
                                               "no suffix")
                        counts[variant][slot, i] = len(bi)
                        bt, dt = torch.as_tensor(bi), torch.as_tensor(di)
                        for j, l in enumerate(gc.LAYERS):
                            c = gc.paired_cos(
                                ps.grads[l][bt].to(probe.device),
                                pp[m].grads[l][dt].to(probe.device))
                            sums[variant][slot, k, i, j] = float(c.sum())
                            means[variant][slot, k, i, j] = float(c.mean())
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{N}", flush=True)
        d = ctx.analysis_dir(self.name)
        arrays = {}
        for v in VARIANTS:
            arrays.update({f"sums|{v}": sums[v], f"means|{v}": means[v],
                           f"counts|{v}": counts[v]})
        gc.save_npz(os.path.join(d, "per_sequence.npz"), **arrays)
        return {"npz": "per_sequence.npz", "n_seq": N,
                "heldout_indices": held, "coefs": list(COEFS),
                "layers": list(gc.LAYERS), "vectors": VARIANTS,
                "member_slots": slots,
                "steering_layers": list(ALL_LAYERS),
                "positions": "tokens after the IP prompt (common suffix)",
                "ip": "bank: cosine vs each bank prompt's gradient, "
                      "token-pooled per prompt, averaged over prompts",
                "token_pooled_mean": {
                    v: token_pooled(sums[v], counts[v]).tolist()
                    for v in VARIANTS},
                "git_commit": gc.git_commit()}


def pooled_table(res: dict, variant: str = "bank") -> np.ndarray:
    """(coef, layer) mean over held-out tokens of cos(g_steer, g_IP)."""
    return np.asarray(res["token_pooled_mean"][variant])


def _prefix(variant: str) -> str:
    return "actdiff" if variant == "bank" else "actdiff.p1"


class IPNotPPSGrad(Product):
    """Section 4.4 figure (Figure 1c before 2026-10-07): per layer,
    cos(grad_IP, grad_actdiff-PPS) vs cos(grad_IP, grad_none). Both are
    cosines between two gradients, so the paper's sign convention does not
    touch them."""

    name = "fig_ipnotpps_grad"
    analyses = [NAME]
    WIDTH = 0.6 * ps.WIDTH_FULL  # printed at 0.6 \\textwidth
    HEIGHT = WIDTH * 3.5 / 6.6   # the old 6.6 x 3.5 aspect

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res = ctx.analysis_result(NAME)
        x = np.asarray(res["layers"])
        ci = res["coefs"].index(ACTDIFF_COEF)
        c0 = res["coefs"].index(0.0)
        t = pooled_table(res)
        fig, ax = ps.figure(self.WIDTH, height=self.HEIGHT)
        # the compared gradient is a PPS gradient (act-diff PPS) or the
        # undefended one, so the curves take the PPS / defenseless roles
        ps.plot_method(ax, x, t[ci], "pps", label="IP vs IP-shaped PPS",
                       marker="o", ms=2.5, end_dot=False)
        ps.plot_baseline(ax, x, t[c0], "IP vs no defense", i="defenseless",
                         markers=True, markevery=1, ms=2.5)
        ax.set_xlabel("layer")
        ax.set_ylabel("gradient cosine")
        ax.set_ylim(0, 1)
        # direct labels in the free space at the left of each curve
        for y, va, dy, text, color, bold in (
                (t[ci][0], "top", -5, "IP vs IP-shaped PPS", ps.PPS, True),
                (t[c0][0], "bottom", 5, "IP vs no defense",
                 ps.baseline_color("defenseless"), False)):
            ax.annotate(text, (x[0], y), xytext=(0, dy),
                        textcoords="offset points", ha="left", va=va,
                        color=color, fontsize=ps.SIZE_SMALL,
                        fontweight="bold" if bold else "normal")
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_ipnotpps_grad"))
        src = f"analysis:{NAME}:token_pooled_mean"
        out = {"actdiff.n_samples": Result(res["n_seq"], decimals=0,
                                           source=src),
               "actdiff.n_coefs": Result(len(res["coefs"]), decimals=0,
                                         source=src)}
        for v in VARIANTS:
            tv, p = pooled_table(res, v), _prefix(v)
            out[f"{p}.gradcos.coef1.mean"] = Result(
                float(tv[ci].mean()), decimals=3, source=f"{src}:{v}")
            out[f"{p}.gradcos.none.mean"] = Result(
                float(tv[c0].mean()), decimals=3, source=f"{src}:{v}")
            out[f"{p}.gradcos.n_layers_below_none"] = Result(
                float(np.sum(tv[ci] < tv[c0])), decimals=0,
                source=f"{src}:{v}")
        return out


class IPvsActdiff(Product):
    """Fig 16: gradient similarity vs steering coefficient and layer."""

    name = "fig_ip_vs_actdiff"
    analyses = [NAME]

    def build(self, ctx: Context) -> Dict[str, Result]:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import BoundaryNorm, ListedColormap

        ps.apply()
        res = ctx.analysis_result(NAME)
        t = pooled_table(res)
        coefs = np.asarray(res["coefs"])
        layers = res["layers"]
        # \\textwidth; the old 14 x 4.6 aspect, a little taller for the
        # colorbars that replace the two legends
        fig, (a, b) = ps.figure("full", height=ps.WIDTH_FULL * 0.36,
                                ncols=2)
        # (a) one line per measured layer: an ordered magnitude that is not
        # a defense. A gray ramp (light -> dark with depth), not viridis:
        # viridis' purple and green ends read as PPS / IP on lines
        lay_cols = ps.plt.get_cmap("Greys")(
            np.linspace(*ps.METHOD_CMAP_RANGE, len(SHOW_LAYERS)))
        for l, col in zip(SHOW_LAYERS, lay_cols):
            a.plot(coefs, t[:, layers.index(l)], color=col, marker="o",
                   ms=2.2, lw=1.2)
        a.set_xlabel("steering intensity")
        a.set_ylabel(r"cos($\nabla\mathcal{L}_{IP}$, "
                     r"$\nabla\mathcal{L}_{PPS}$)")
        bounds = np.arange(len(SHOW_LAYERS) + 1) - 0.5
        cb = ps.colorbar(fig, ScalarMappable(
            BoundaryNorm(bounds, len(SHOW_LAYERS)),
            ListedColormap(lay_cols)), a, label="layer")
        cb.set_ticks(range(len(SHOW_LAYERS)), labels=[str(l) for l in
                                                      SHOW_LAYERS])
        # (b) one line per signed intensity of act-diff PPS: grays below
        # zero, the PPS ramp above; 0 (no defense) in the defenseless style
        cmap = ps.signed_sweep_cmap("pps")
        norm = ps.diverging_norm(vmax=float(np.abs(coefs).max()))
        for k, c in enumerate(coefs):
            if c == 0:
                ps.plot_baseline(b, layers, t[k], "0 (no defense)",
                                 i="defenseless", zorder=4)
            else:
                b.plot(layers, t[k], color=cmap(norm(c)), lw=1.2,
                       marker="o", ms=1.8)
        b.set_xlabel("layer")
        # the lines bunch at both ends: key the baseline in the empty top
        # left instead of labeling it directly
        b.legend(loc="upper left", handlelength=2.4, borderaxespad=0.2)
        ps.colorbar(fig, ScalarMappable(norm, cmap), b, label="intensity")
        for ax, letter in ((a, "a"), (b, "b")):
            ps.panel_label(ax, letter)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_ip_vs_actdiff"))
        out = {}
        src = f"analysis:{NAME}"
        for v in VARIANTS:
            tv, p = pooled_table(res, v), _prefix(v)
            best = np.unravel_index(np.argmax(tv), tv.shape)
            out[f"{p}.gradcos.max"] = Result(float(tv[best]), decimals=3,
                                             source=f"{src}:{v}")
            out[f"{p}.gradcos.argmax_coef"] = Result(
                float(coefs[best[0]]), decimals=2, source=f"{src}:{v}")
        return out


class ActdiffVecCos(Product):
    """Fig 17: cos(act-diff vector, evil PV) per layer (from the files)."""

    name = "fig_actdiff_vec_cos"

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        e = gc.load_vector_file(EVIL)
        layers = list(gc.LAYERS)
        # printed at 0.75 \\textwidth; the old 7.5 x 3.8 aspect
        w = 0.75 * ps.WIDTH_FULL
        fig, ax = ps.figure(w, height=w * 3.8 / 7.5)
        out: Dict[str, Result] = {}
        for v, path in VARIANTS.items():
            vec = gc.load_vector_file(path)
            cs = np.array([gc.cos(vec[l], e[l]) for l in layers])
            # vectors built from the IP prompts: IP's color, p1 its lighter
            # ramp variant (dashed as well, for grayscale)
            ps.plot_method(ax, layers, cs, "ip", variant=0 if v == "bank"
                           else 1, marker="o", ms=2.5, end_dot=False,
                           ls="-" if v == "bank" else (0, (4, 2)),
                           lw=1.9 if v == "bank" else 1.5,
                           label="IP (3 prompts)" if v == "bank"
                           else "IP (p1 only)")
            meta = actdiff_meta(path)
            src, p = f"file:{path}", _prefix(v)
            out.update({
                f"{p}.veccos.l20": Result(float(cs[layers.index(20)]),
                                          decimals=2, source=src),
                f"{p}.veccos.max": Result(float(cs.max()), decimals=2,
                                          source=src),
                f"{p}.veccos.min": Result(float(cs.min()), decimals=2,
                                          source=src)})
            if v == "bank":
                out["actdiff.n_build"] = Result(meta["split"]["n_build"],
                                                decimals=0, source=src)
                out["actdiff.n_heldout"] = Result(meta["split"]["n_heldout"],
                                                  decimals=0, source=src)
        ps.zero_line(ax)
        ax.set_xlabel("layer")
        ax.set_ylabel('cos(act-diff vector, "evil" PV)')
        ps.legend_above(ax)  # the two lines overlap: no room to label
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_actdiff_vec_cos"))
        return out


class ActdiffBehavior(Product):
    """Section 4.4: act-diff PPS vs IP evil expression (base x heavy), for
    IP (3 prompts) and the single-prompt comparison."""

    name = "sec_actdiff_behavior"
    runs = [RUN, IP_RUN, RUN_P1, IP_P1_RUN]

    def build(self, ctx: Context) -> Dict[str, Result]:
        out = {"sec4.4.actdiff.coef": Result(ACTDIFF_COEF, decimals=0,
                                             source="gradient_actdiff."
                                                    "ACTDIFF_COEF"),
               "sec4.4.actdiff.n_layers": Result(len(ALL_LAYERS), decimals=0,
                                                 source="runs.ALL_LAYERS")}
        for tag, run, ip in (("", RUN, IP_RUN), (".p1", RUN_P1, IP_P1_RUN)):
            out[f"sec4.4{tag}.actdiff.evil"] = ctx.trait_score(run)
            out[f"sec4.4{tag}.ip.evil"] = ctx.trait_score(ip)
            out[f"sec4.4{tag}.actdiff.coherence"] = ctx.trait_score(
                run, column="coherence")
            out[f"sec4.4{tag}.ip.coherence"] = ctx.trait_score(
                ip, column="coherence")
        return out
