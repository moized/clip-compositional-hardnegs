"""compbind: compositional binding vs. text-only shortcuts in CLIP.

Code for the YZM5110 research project. See README.md for the experiment plan
and docs/CODE_TOUR.md for a map of the modules.
"""

__version__ = "0.1.0"

# Training conditions = the kind of negative caption used during fine-tuning.
#   none         plain CLIP fine-tuning, no extra negatives           (control, "CLIP-FT")
#   random       one caption of another image as an extra negative    (control for "more negatives")
#   swap         NegCLIP-style swapped captions                       (reproduced baseline)
#   swap_nosym   swapped captions without meaning-preserving swaps    (ablation)
#   swap_fluent  fluency-matched swapped captions                     (the improvement)
CONDITIONS = ("none", "random", "swap", "swap_nosym", "swap_fluent")
