"""Zero-shot classification and retrieval metrics, with stand-in models and tiny fake datasets."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compbind import downstream  # noqa: E402
from compbind.models import ClipBundle  # noqa: E402


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")


def test_imagenet_files_kaggle_layout(tmp_path):
    root = str(tmp_path)
    with open(os.path.join(root, "LOC_synset_mapping.txt"), "w") as f:
        f.write("n00000001 first class\nn00000002 second class\nn00000003 third class\n")
    with open(os.path.join(root, "LOC_val_solution.csv"), "w") as f:
        f.write("ImageId,PredictionString\nval_2,n00000003 1 2 3 4\nval_1,n00000001 5 6 7 8 n00000002 1 1 2 2\n")
    files = downstream.imagenet_files(root, n_classes=3)
    assert [(os.path.basename(p), label) for p, label in files] == [("val_1.JPEG", 0), ("val_2.JPEG", 2)]


def test_imagenet_files_one_folder_per_class(tmp_path):
    for class_id, name in (("n00000002", "b.JPEG"), ("n00000001", "a.JPEG"), ("n00000001", "c.JPEG")):
        touch(tmp_path / "val" / class_id / name)
    files = downstream.imagenet_files(str(tmp_path), n_classes=2)
    assert [label for _, label in files] == [0, 0, 1]


def test_imagenet_files_rejects_wrong_folders(tmp_path):
    with pytest.raises(FileNotFoundError):
        downstream.imagenet_files(str(tmp_path))
    touch(tmp_path / "val" / "n00000001" / "a.JPEG")
    with pytest.raises(ValueError):
        downstream.imagenet_files(str(tmp_path))          # 1 class instead of 1000


class FakeClip:
    """Text "class k" and image tensors filled with k both map to the k-th unit vector."""

    def eval(self):
        return self

    def encode_text(self, tokens):
        return torch.nn.functional.one_hot(tokens[:, 0], 4).float()

    def encode_image(self, images):
        return torch.nn.functional.one_hot(images.view(len(images), -1)[:, 0].long(), 4).float()


def fake_bundle():
    tokenizer = lambda prompts: torch.tensor([[int(p.split()[-1])] for p in prompts])  # noqa: E731
    return ClipBundle(FakeClip(), None, None, tokenizer, "fake", "fake", "cpu")


def test_zero_shot_accuracy(monkeypatch):
    labels = [0, 1, 2, 3, 3, 2]
    shown = [0, 1, 2, 3, 0, 2]                               # the fifth image looks like class 0
    data = [(torch.full((1, 2, 2), float(s)), y) for s, y in zip(shown, labels)]
    monkeypatch.setattr(downstream, "build_zero_shot_dataset",
                        lambda name, root, transform, limit=None, seed=0:
                        (data, ["0", "1", "2", "3"], ["class {c}", "a photo of class {c}"]))
    out = downstream.zero_shot(fake_bundle(), "fake", "unused", batch_size=4, workers=0)
    assert out["acc1"] == pytest.approx(5 / 6)
    assert out["acc5"] == 1.0                                # only 4 classes, so top-5 covers all
    assert out["n_images"] == 6 and out["n_classes"] == 4 and out["n_templates"] == 2


def test_class_vectors_are_normalised():
    vectors = downstream.class_vectors(fake_bundle(), ["0", "2"], ["class {c}", "photo of {c}"])
    assert vectors.shape == (2, 4)
    assert torch.allclose(vectors.norm(dim=1), torch.ones(2))


class FakeEmbedder:
    def __init__(self, image_vectors, text_vectors):
        self.image_vectors, self.text_vectors = image_vectors, text_vectors

    def images(self, keys):
        return np.asarray(self.image_vectors, dtype=np.float32)

    def texts(self, texts):
        return np.asarray(self.text_vectors, dtype=np.float32)


class FakeImages:
    def path(self, image_id):
        return f"image{image_id}"


def test_coco_retrieval_recall():
    test_split = [{"image": "val2014/COCO_val2014_000000000001.jpg", "caption": ["a", "b"]},
                  {"image": "val2014/COCO_val2014_000000000002.jpg", "caption": ["c", "d"]}]
    e0, e1 = [1.0, 0.0], [0.0, 1.0]
    # captions a, b, c match their image; caption d points at the wrong image
    out = downstream.coco_retrieval(FakeEmbedder([e0, e1], [e0, e0, e1, e0]), test_split, FakeImages())
    assert out["image_r1"] == pytest.approx(3 / 4)           # text -> image
    assert out["text_r1"] == 1.0                             # image -> text: each image finds a true caption
    assert out["n_images"] == 2 and out["n_texts"] == 4
