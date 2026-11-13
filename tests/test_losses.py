"""The contrastive losses, checked against hand-computed values."""
import math

import pytest

torch = pytest.importorskip("torch")

from compbind.losses import batch_diagnostics, clip_loss, negclip_loss  # noqa: E402


def unit(rows):
    return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def test_clip_loss_matches_manual_cross_entropy():
    images = unit([[1.0, 0.0], [0.0, 1.0]])
    texts = unit([[1.0, 0.2], [0.1, 1.0]])
    scale = torch.tensor(5.0)
    logits = scale * images @ texts.t()
    rows = -(torch.log_softmax(logits, dim=1).diag()).mean()
    columns = -(torch.log_softmax(logits, dim=0).diag()).mean()
    assert clip_loss(images, texts, scale).item() == pytest.approx(((rows + columns) / 2).item(), abs=1e-6)


def test_negclip_loss_uses_negatives_only_for_images():
    images = unit([[1.0, 0.0], [0.0, 1.0]])
    texts = unit([[1.0, 0.0], [0.0, 1.0]])
    scale = torch.tensor(10.0)
    # A negative identical to the true caption: each image has two equally good captions.
    loss = negclip_loss(images, texts, texts.clone(), scale).item()
    image_to_text = math.log(2 + 2 * math.exp(-10.0))        # two logits of 10, two logits of 0
    text_to_image = math.log(1 + math.exp(-10.0))            # negatives are not in this term
    assert loss == pytest.approx((image_to_text + text_to_image) / 2, abs=1e-5)


def test_negclip_loss_equals_clip_loss_when_negatives_are_irrelevant():
    generator = torch.Generator().manual_seed(0)
    images = torch.nn.functional.normalize(torch.randn(6, 16, generator=generator), dim=-1)
    texts = torch.nn.functional.normalize(images + 0.1 * torch.randn(6, 16, generator=generator), dim=-1)
    far = -images                                             # cosine -1 with their own image
    scale = torch.tensor(50.0)
    assert negclip_loss(images, texts, far, scale).item() == pytest.approx(
        clip_loss(images, texts, scale).item(), abs=1e-4)


def test_hard_negative_increases_the_loss_and_has_gradient():
    generator = torch.Generator().manual_seed(1)
    images = torch.nn.functional.normalize(torch.randn(4, 8, generator=generator), dim=-1)
    texts = images.clone().requires_grad_(True)
    negatives = torch.nn.functional.normalize(images + 0.05 * torch.randn(4, 8, generator=generator), dim=-1)
    negatives.requires_grad_(True)
    scale = torch.tensor(30.0)
    loss = negclip_loss(images, texts, negatives, scale)
    assert loss.item() > clip_loss(images, texts, scale).item()
    loss.backward()
    assert negatives.grad.abs().sum() > 0


def test_batch_diagnostics():
    images = unit([[1.0, 0.0], [0.0, 1.0]])
    texts = unit([[1.0, 0.1], [0.1, 1.0]])
    negatives = unit([[1.0, 0.0], [1.0, 0.0]])    # first negative beats the caption, second does not
    out = batch_diagnostics(images, texts, negatives)
    assert out["retrieval_acc"] == 1.0
    assert out["pair_acc"] == 0.5
