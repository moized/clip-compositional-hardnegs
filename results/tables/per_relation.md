### VG-Relation by relation: the twelve most frequent relations

| Relation | Items | Share of all items (%) | In ARO's score | CLIP | NegCLIP | Text only: GPT-2 |
|---|---|---|---|---|---|---|
| to the right of | 7741 | 32.3 | yes | 49.6 | 50.0 | 49.7 |
| to the left of | 7741 | 32.3 | yes | 49.7 | 49.4 | 50.1 |
| on | 1684 | 7.0 | yes | 52.6 | 90.9 | 87.3 |
| wearing | 949 | 4.0 | yes | 47.3 | 99.8 | 99.9 |
| in | 708 | 3.0 | yes | 62.9 | 93.4 | 80.6 |
| near | 689 | 2.9 | no | 49.6 | 49.6 | 49.6 |
| in front of | 588 | 2.5 | yes | 54.1 | 76.9 | 62.8 |
| behind | 574 | 2.4 | yes | 56.1 | 29.4 | 44.1 |
| of | 367 | 1.5 | no | 43.6 | 40.1 | 62.4 |
| above | 269 | 1.1 | yes | 47.6 | 59.5 | 60.2 |
| below | 209 | 0.9 | yes | 56.0 | 39.7 | 49.3 |
| next to | 203 | 0.8 | no | 48.8 | 48.3 | 53.2 |
| Mean over the 48 relations in ARO's score (macro accuracy) | 21944 | 91.7 |  | 59.9 | 81.8 | 82.8 |
| All items (per-item accuracy) | 23937 | 100.0 |  | 51.1 | 58.7 | 58.8 |

- Accuracy in %; chance is 50%. NegCLIP is the released checkpoint with QuickGELU.
- ARO's score gives every relation the same weight, whatever its number of items; the last two rows show how much this choice matters.
