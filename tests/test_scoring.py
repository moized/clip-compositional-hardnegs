"""Scoring test items with a stand-in model (no CLIP weights needed)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from compbind.benchmarks import items_fingerprint, summarize  # noqa: E402
from compbind.scoring import Item, score_items, shuffled_image_index  # noqa: E402


class FakeEmbedder:
    """Embeds an image as a one-hot vector for its "content" and a caption likewise."""

    def __init__(self, image_content, text_content, dim=8):
        self.image_content, self.text_content, self.dim = image_content, text_content, dim

    def _one_hot(self, indices):
        out = np.zeros((len(indices), self.dim), dtype=np.float32)
        out[np.arange(len(indices)), indices] = 1.0
        return out

    def images(self, keys):
        return self._one_hot([self.image_content[path] for path, _ in keys])

    def texts(self, texts):
        return self._one_hot([self.text_content[t] for t in texts])


def make_items():
    # image "img0" shows content 0, "img1" shows content 1, "img2" shows content 2
    return [
        Item("a/0", "img0", ("wrong caption", "caption zero"), 1, "g1", "0"),
        Item("a/1", "img1", ("caption zero", "caption one"), 1, "g1", "1"),
        Item("a/2", "img2", ("caption two", "caption zero"), 0, "g2", "2"),
        Item("a/3", "img2", ("caption one", "wrong caption"), 0, "g2", "2"),   # neither caption matches
    ]


def fake_embedder():
    return FakeEmbedder({"img0": 0, "img1": 1, "img2": 2},
                        {"caption zero": 0, "caption one": 1, "caption two": 2, "wrong caption": 7})


def test_score_items_predictions_and_margins():
    table = score_items(fake_embedder(), make_items())
    assert table["correct"].tolist() == [1, 1, 1, 1]          # the tie in the last item goes to index 0
    assert table["pred"].tolist() == [1, 1, 0, 0]
    assert table["margin"].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0])
    assert {"s0", "s1", "correct_shuffled", "pred_shuffled"} <= set(table.columns)


def test_tie_counts_as_error_when_the_true_caption_is_listed_second():
    items = [Item("b/0", "img2", ("wrong caption", "caption one"), 1, "g", "2"),
             Item("b/1", "img0", ("wrong caption", "caption zero"), 1, "g", "0")]
    table = score_items(fake_embedder(), items)
    assert table["correct"].tolist() == [0, 1]                # equal scores -> first candidate -> wrong


def test_shuffled_image_never_keeps_the_own_image():
    image_ids = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"])
    partner = shuffled_image_index(image_ids, seed=0)
    assert (image_ids[partner] != image_ids).all()
    assert (shuffled_image_index(image_ids, seed=0) == partner).all()


def test_shuffled_image_needs_two_images():
    with pytest.raises(ValueError):
        shuffled_image_index(np.array(["a", "a"]))


def test_fingerprint_changes_with_the_captions():
    items = make_items()
    changed = items[:-1] + [Item("a/3", "img2", ("caption one", "another caption"), 0, "g2", "2")]
    assert items_fingerprint(items) == items_fingerprint(make_items())
    assert items_fingerprint(items) != items_fingerprint(changed)


def test_summarize_sugarcrepe_uses_the_mean_of_subsets():
    import pandas as pd

    table = pd.DataFrame({"group": ["swap_obj"] * 2 + ["add_att"] * 8,
                          "image_id": [str(i) for i in range(10)],
                          "correct": [1, 1] + [0] * 8})
    out = summarize("sugarcrepe", table, n_boot=100)
    assert out["headline"] == pytest.approx(0.5)              # (1.0 + 0.0) / 2
    assert out["overall_pooled"]["value"] == pytest.approx(0.2)
    assert out["categories"]["swap"]["pooled"]["value"] == pytest.approx(1.0)


def test_summarize_vg_relation_drops_symmetric_relations():
    import pandas as pd

    table = pd.DataFrame({"group": ["on"] * 4 + ["next to"] * 4,     # "next to" is in ARO's symmetric list
                          "image_id": [str(i) for i in range(8)],
                          "correct": [1, 1, 1, 0, 0, 0, 0, 0]})
    out = summarize("vg_relation", table, n_boot=100)
    assert out["headline"] == pytest.approx(0.75)
    assert out["micro_accuracy"]["value"] == pytest.approx(3 / 8)
    assert out["n_groups"] == 1


def test_training_pairs_use_the_reduced_copy_when_it_exists(tmp_path):
    import pandas as pd

    from compbind.coco import small_copy_path
    from compbind.training import PairDataset

    class Originals:
        def path(self, image_id):
            return f"original-{image_id}.jpg"

    folder = str(tmp_path)
    open(small_copy_path(folder, 7), "wb").close()
    table = pd.DataFrame({"image_id": [7, 8], "caption": ["a", "b"]})
    dataset = PairDataset(table, Originals(), None, "none", small_dir=folder)
    assert dataset.paths == [small_copy_path(folder, 7), "original-8.jpg"]
    assert dataset.n_small == 1
    assert PairDataset(table, Originals(), None, "none").paths == ["original-7.jpg", "original-8.jpg"]
