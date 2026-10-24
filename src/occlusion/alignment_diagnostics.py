"""Render rejected alignment evidence and simulate pixel-selection sensitivity.

This writes diagnostics only, never a geometry configuration. Pixel perturbations
are a chosen sensitivity experiment, not an estimate of real click uncertainty.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .alignment import align_correspondences, load_pair, pair_identity, sample_point


def jitter_pixels(pixels, shape, radius, rng):
    """Independent uniform integer offsets in [-radius,+radius] per coordinate.

    Out-of-image trials are invalid rather than clamped to a boundary pixel.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Pixel radius must be a nonnegative integer")
    shifted = np.asarray(pixels, dtype=int) + rng.integers(-radius, radius+1, np.asarray(pixels).shape)
    if ((shifted < 0).any() or (shifted[:, 0] >= shape[1]).any()
            or (shifted[:, 1] >= shape[0]).any()):
        raise ValueError("Perturbed pixels lie outside the image")
    return shifted


def fixed_span_pass(errors, training, validation, span, inlier_fraction, validation_fraction):
    """Check a trial against the original absolute tolerances and fixed split."""
    required = max(6, int(np.ceil(0.75 * len(training))))
    return bool(np.sum(errors[training] <= inlier_fraction*span) >= required
                and np.max(errors[validation]) <= validation_fraction*span)


def inspect_alignment(pair, matches_path, report_path, trials=100, seed=42):
    if not isinstance(trials, int) or trials < 1:
        raise ValueError("Trials must be a positive integer")
    pair = Path(pair)
    matches = json.loads(Path(matches_path).read_text())
    baseline = json.loads(Path(report_path).read_text())
    identity = pair_identity(pair)
    if matches["capture_identity"] != identity or baseline["capture_identity"] != identity:
        raise ValueError("Matches and report must reference this exact saved pair")
    results = load_pair(pair)
    pixels = {name: np.asarray([item[name] for item in matches["pairs"]], dtype=int) for name in results}
    points = {name: np.array([sample_point(results[name], uv) for uv in pixels[name]]) for name in results}
    scale, rotation, translation, check = align_correspondences(
        points["wrist"], points["overhead"], baseline["inlier_threshold_fraction"], baseline["validation_threshold_fraction"])
    if not np.allclose(check["per_pair_errors_relative"], baseline["per_pair_errors_relative"], rtol=1e-7, atol=1e-10):
        raise ValueError("Report does not reproduce from these matches; regenerate it first")
    fitted = scale*points["wrist"] @ rotation.T + translation
    span = baseline["training_target_extent_relative"]
    training, validation = np.array(baseline["training_indices_zero_based"]), np.array(baseline["validation_indices_zero_based"])
    inlier_fraction, validation_fraction = baseline["inlier_threshold_fraction"], baseline["validation_threshold_fraction"]
    projected = fitted @ np.asarray(results["overhead"].intrinsics).T
    projected = projected[:, :2] / projected[:, 2, None]
    per_point = []
    for i in range(len(fitted)):
        moves = {}
        for name in results:
            movements = []
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                try:
                    shifted = sample_point(results[name], pixels[name][i] + [du, dv])
                    movements.append(float(np.linalg.norm(shifted-points[name][i]) * (scale if name == "wrist" else 1) / span))
                except ValueError:
                    continue
            moves[name] = movements
        per_point.append({
            "point_number": i+1, "role": "held_out" if i in validation else "training",
            "error_fraction_of_matched_span": float(np.linalg.norm(fitted[i]-points["overhead"][i])/span),
            "fitted_wrist_reprojection_error_overhead_px": float(np.linalg.norm(projected[i]-pixels["overhead"][i])),
            "axial_depth_error_fraction_of_overhead_depth": float((fitted[i,2]-points["overhead"][i,2])/points["overhead"][i,2]),
            "one_pixel_neighbor_displacement_fraction_of_span": moves,
        })
    rng = np.random.default_rng(seed)
    experiments = []
    for radius in (0, 1, 2, 3):
        records = []
        for _ in range(1 if radius == 0 else trials):
            try:
                moved = {name: jitter_pixels(pixels[name], results[name].depth.shape, radius, rng) for name in results}
                p = {name: np.array([sample_point(results[name], uv) for uv in moved[name]]) for name in results}
                # Fixed transform isolates click/depth-sampling changes from fit changes.
                fixed_errors = np.linalg.norm(scale*p["wrist"]@rotation.T+translation-p["overhead"], axis=1)
                trial_scale, trial_rotation, trial_translation, report = align_correspondences(
                    p["wrist"], p["overhead"], inlier_fraction, validation_fraction)
                errors = np.asarray(report["per_pair_errors_relative"])
                angle = np.degrees(np.arccos(np.clip((np.trace(trial_rotation@rotation.T)-1)/2, -1, 1)))
                records.append({
                    "valid": True,
                    "fixed_transform_max_heldout_error_fraction_original_span": float(np.max(fixed_errors[validation])/span),
                    "refitted_max_heldout_error_fraction_original_span": float(np.max(errors[validation])/span),
                    "refitted_max_heldout_error_fraction_trial_span": report["validation_max_error_normalized"],
                    "original_span_pass": fixed_span_pass(errors, training, validation, span, inlier_fraction, validation_fraction),
                    "existing_code_pass": report["accepted"],
                    "scale_ratio_to_original_fit": trial_scale/scale,
                    "rotation_change_degrees": float(angle),
                    "trial_span_ratio_to_original": report["training_target_extent_relative"]/span,
                })
            except (ValueError, np.linalg.LinAlgError) as error:
                records.append({"valid": False, "reason": str(error)})
        valid = [r for r in records if r["valid"]]
        summary = {"pixel_radius": radius, "attempted_trials": len(records), "valid_trials": len(valid),
                   "original_span_pass_count": sum(r["original_span_pass"] for r in valid),
                   "existing_code_pass_count": sum(r["existing_code_pass"] for r in valid)}
        for field in ("fixed_transform_max_heldout_error_fraction_original_span",
                      "refitted_max_heldout_error_fraction_original_span", "scale_ratio_to_original_fit",
                      "rotation_change_degrees", "trial_span_ratio_to_original"):
            summary[field+"_p05_p50_p95"] = np.percentile([r[field] for r in valid], [5,50,95]).tolist() if valid else None
        experiments.append({"summary": summary, "trials": records})
        print(f"Radius {radius}px: {summary['original_span_pass_count']}/{len(records)} trials pass original-span gates", flush=True)
    return {
        "schema_version": 1, "pair": str(pair), "matches": str(matches_path), "source_report": str(report_path),
        "capture_identity": identity, "seed": seed, "original_span_relative_units": span,
        "inlier_fraction": inlier_fraction, "validation_fraction": validation_fraction,
        "per_point": per_point, "experiments": experiments,
        "interpretation": ["Synthetic click jitter is not a measured uncertainty distribution",
                           "Held-out matches never participate in transform fitting",
                           "Original-span checks keep absolute error tolerances fixed across trials",
                           "Existing-code checks recompute matched span; changes in span can change acceptance",
                           "No trial is selected as a calibration and no geometry file is written",
                           "Results do not isolate depth errors from inferred intrinsics or point-selection errors"],
    }, points, fitted, pixels, results


def plot_diagnostics(output, report, points, fitted, pixels, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = points["overhead"]
    center = target.mean(axis=0)
    _, _, basis = np.linalg.svd(target-center, full_matrices=False)
    reference, transformed = (target-center)@basis.T, (fitted-center)@basis.T
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for ax, dims, title in ((axes[0,0], (0,1), "Along the selected overhead surface"),
                            (axes[0,1], (0,2), "Side view of the selected overhead surface")):
        a, b = dims
        ax.scatter(reference[:,a], reference[:,b], c="#1676b5", s=45, label="Overhead observed points")
        ax.scatter(transformed[:,a], transformed[:,b], c="#e58226", marker="x", s=55, label="Wrist points after fitted transform")
        for i in range(len(target)):
            ax.plot([reference[i,a], transformed[i,a]], [reference[i,b], transformed[i,b]], color="gray", lw=1)
            ax.annotate(str(i+1), reference[i,[a,b]], xytext=(4,4), textcoords="offset points", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"Display axis {a+1} (relative units)"); ax.set_ylabel(f"Display axis {b+1} (relative units)")
        ax.set_title(title); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    ax=axes[1,0]
    point_errors=np.array([p["error_fraction_of_matched_span"] for p in report["per_point"]])
    heldout=np.array([p["role"] == "held_out" for p in report["per_point"]])
    colors=np.where(heldout,"#b34388","#748b99")
    ax.bar(np.arange(1,len(target)+1),100*point_errors,color=colors)
    ax.axhline(100*report["inlier_fraction"],color="#555555",ls=":",label="Training inlier limit")
    ax.axhline(100*report["validation_fraction"],color="#b34388",ls="--",label="Held-out limit")
    ax.set_xticks(np.arange(1,len(target)+1)); ax.set_xlabel("Point number (purple = held out)")
    ax.set_ylabel("Error / original matched span (%)"); ax.set_title("Original fit: point-by-point disagreement"); ax.legend(fontsize=8)
    ax=axes[1,1]
    labels=[]; series=[]
    for experiment in report["experiments"]:
        if experiment["summary"]["pixel_radius"] == 0:
            continue
        values=[100*t["refitted_max_heldout_error_fraction_original_span"] for t in experiment["trials"] if t["valid"]]
        if values:
            series.append(values); labels.append(f"+/- {experiment['summary']['pixel_radius']} px")
    if series:
        ax.boxplot(series,tick_labels=labels,showfliers=False)
    ax.axhline(100*report["validation_fraction"],color="#b34388",ls="--",label="Original held-out limit")
    ax.set_ylabel("Worst held-out error / original span (%)")
    ax.set_title("Sensitivity after refitting each perturbed click set\nBoxes: median and quartiles; outliers hidden"); ax.legend(fontsize=8)
    fig.suptitle("Alignment evidence: matched points, fitted discrepancies, and pixel sensitivity\nDisplay axes are PCA of overhead points; no geometry correction or metric ground truth",fontsize=14)
    fig.savefig(output/"alignment-evidence.png",dpi=160)
    plt.close(fig)
    # Image crop for matching the point numbers to the user's actual features.
    fig, axes=plt.subplots(1,2,figsize=(14,6),layout="constrained")
    for ax,name in zip(axes,("wrist","overhead")):
        ax.imshow(results[name].rgb)
        uv=pixels[name]
        for i,(u,v) in enumerate(uv):
            ax.scatter(u,v,s=35,facecolors="none",edgecolors="#00ff80")
            ax.annotate(str(i+1),(u,v),xytext=(5,-4),textcoords="offset points",color="white",fontsize=10,
                        bbox=dict(facecolor="black",alpha=.7,pad=1,edgecolor="none"))
        ax.set_xlim(uv[:,0].min()-20,uv[:,0].max()+20)
        ax.set_ylim(uv[:,1].max()+20,uv[:,1].min()-20)
        ax.set_title(name+" saved processed RGB"); ax.set_xlabel("Column"); ax.set_ylabel("Row")
    fig.suptitle("Revised matches: exact selected pixels, enlarged for inspection")
    fig.savefig(output/"selected-points.png",dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair",type=Path,required=True)
    parser.add_argument("--matches",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trials",type=int,default=100)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output directory already exists; choose a new directory")
        report,points,fitted,pixels,results=inspect_alignment(args.pair,args.matches,args.report,args.trials,args.seed)
        args.output.mkdir(parents=True)
        (args.output/"sensitivity.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        plot_diagnostics(args.output,report,points,fitted,pixels,results)
        print(f"Saved diagnostics to {args.output}; no alignment configuration written")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1,f"Alignment diagnostics: {error}\n")


if __name__ == "__main__":
    main()
