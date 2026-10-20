"""What each wafer-map pattern usually points at on the fab floor.

A classifier only says *which* pattern a wafer shows. The value to a yield
engineer is the next question: which process step or tool to look at. These
are the commonly cited first suspects for each spatial signature — starting
points for an investigation, not diagnoses. The real cause is confirmed by
correlating the pattern with lot history, tool/chamber IDs and in-line
metrology.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signature:
    pattern: str
    looks_like: str
    first_suspects: tuple[str, ...]
    check_next: str


SIGNATURES: dict[str, Signature] = {
    s.pattern: s
    for s in (
        Signature(
            "Center",
            "Failures clustered around the wafer centre.",
            (
                "Radial non-uniformity in a process that runs centre-out: spin-coat or develop puddle, "
                "CMP pressure profile, gas or heat distribution in a chamber",
                "Film-thickness or etch-rate variation peaking at the centre",
            ),
            "Compare centre vs. edge thickness/CD metrology for the lot; check which chambers processed it.",
        ),
        Signature(
            "Donut",
            "A ring of failures at mid-radius, with a cleaner centre and edge.",
            (
                "Zone-to-zone mismatch in a multi-zone heater, chuck or CMP carrier",
                "Deposition or etch uniformity problem at a specific radius",
            ),
            "Look at radial uniformity maps from the deposition/etch/CMP steps; a fixed radius suggests hardware zoning.",
        ),
        Signature(
            "Edge-Ring",
            "Failures all the way round the wafer edge.",
            (
                "Edge exclusion effects: film roll-off, edge-bead removal, etch-rate change at the edge",
                "Lithography focus or exposure drift near the edge; wafer bow",
            ),
            "Check edge-die metrology and edge-bead / bevel-clean steps; a full ring means a rotationally symmetric cause.",
        ),
        Signature(
            "Edge-Loc",
            "A localised failure cluster at one part of the edge.",
            (
                "Mechanical contact: clamp ring, chuck pins, robot end-effector, cassette slot",
                "Local edge chipping or contamination introduced in handling",
            ),
            "Note the angular position relative to the notch; a fixed position across wafers points to a specific handler or tool.",
        ),
        Signature(
            "Loc",
            "A localised failure cluster in the interior.",
            (
                "Particle or contamination event during a process step",
                "Local process excursion (e.g. a reticle or mask defect if it repeats per field)",
            ),
            "Overlay with in-line defect-inspection maps for the same wafer to find the step where the cluster first appears.",
        ),
        Signature(
            "Scratch",
            "A thin linear or arc-shaped trail of failures.",
            (
                "Mechanical scratching, classically CMP (pad debris, agglomerated slurry particles)",
                "Wafer handling or transfer damage",
            ),
            "Check CMP consumables and pad conditioning for the lot; arcs that follow the polish path are a strong CMP signal.",
        ),
        Signature(
            "Random",
            "Failures scattered across the wafer with no spatial structure.",
            (
                "Random defect density — airborne or chemical particles, material quality",
                "Below a threshold this is the normal yield baseline, not an excursion",
            ),
            "Trend random-defect density over time and by tool; a rise without spatial structure suggests environment or materials.",
        ),
        Signature(
            "Near-full",
            "Nearly every die on the wafer fails.",
            (
                "Gross process failure: wrong recipe, missed or doubled step, tool fault",
                "Test-side problem: probe card, prober setup or test program",
            ),
            "Rule out the tester first (re-probe a sample), then check the wafer's process history for a missed step.",
        ),
        Signature(
            "none",
            "No systematic failure pattern.",
            ("Normal yield loss",),
            "No action from the map alone.",
        ),
    )
}


def explain(pattern: str) -> Signature:
    try:
        return SIGNATURES[pattern]
    except KeyError:
        raise KeyError(f"no signature for pattern {pattern!r}") from None
