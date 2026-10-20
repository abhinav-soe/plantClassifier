import numpy as np
import pytest
import torch
from PIL import Image

from vision.data import eval_transform, get_datasets, stratified_split
from vision.flower_names import FLOWER_NAMES
from vision.gradcam import GradCAM, overlay
from vision.model import ARCHS, build_model, gradcam_layer, set_backbone_trainable, split_params
from vision.predict import explain, load_checkpoint, save_checkpoint, get_device

SIZE = 64


def test_flower_names_complete():
    assert len(FLOWER_NAMES) == 102 == len(set(FLOWER_NAMES)) and FLOWER_NAMES[53] == "sunflower"


@pytest.mark.parametrize("arch", ARCHS)
def test_model_output_and_gradcam_layer(arch):
    m = build_model(arch, 5, pretrained=False).eval()
    assert m(torch.randn(2, 3, SIZE, SIZE)).shape == (2, 5)
    assert gradcam_layer(m, arch) is not None


def test_freeze_only_trains_head():
    m = build_model("resnet18", 4, pretrained=False)
    set_backbone_trainable(m, "resnet18", False)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}
    set_backbone_trainable(m, "resnet18", True)
    backbone, head = split_params(m, "resnet18")
    assert all(p.requires_grad for p in backbone + head) and len(head) == 2


@pytest.mark.parametrize("arch", ["resnet18", "efficientnet_b0"])
def test_gradcam_map_valid_and_class_specific(arch):
    torch.manual_seed(0)
    m = build_model(arch, 6, pretrained=False)
    # freeze everything: Grad-CAM must still work because the *input* requires grad
    for p in m.parameters():
        p.requires_grad = False
    cam = GradCAM(m, gradcam_layer(m, arch))
    x = torch.randn(1, 3, SIZE, SIZE)
    c0, p0, i0 = cam(x, 0)
    c1, _, i1 = cam(x, 3)
    cam.remove()
    assert c0.shape == (SIZE, SIZE) and 0 <= c0.min() and c0.max() <= 1 + 1e-6 and not np.isnan(c0).any()
    assert (i0, i1) == (0, 3) and abs(p0.sum() - 1) < 1e-4
    assert not np.allclose(c0, c1)                      # different target class -> different explanation


def test_overlay_shape_and_dtype():
    out = overlay(np.zeros((SIZE, SIZE, 3), dtype="uint8"), np.random.rand(SIZE, SIZE))
    assert out.shape == (SIZE, SIZE, 3) and out.dtype == np.uint8


def test_stratified_split_is_disjoint_and_balanced():
    y = np.repeat(np.arange(4), 50)
    tr, va, te = stratified_split(y)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == 200 and set(y[te]) == set(range(4))


@pytest.fixture
def shapes_dir(tmp_path):
    from PIL import ImageDraw
    rng = np.random.default_rng(0)
    for cls, color in [("red", (220, 30, 30)), ("green", (30, 200, 30)), ("blue", (30, 30, 220))]:
        (tmp_path / cls).mkdir()
        for i in range(24):
            im = Image.fromarray(rng.integers(0, 60, (96, 96, 3), dtype="uint8"))
            x, y = rng.integers(10, 40, 2)
            ImageDraw.Draw(im).ellipse([x, y, x + 40, y + 40], fill=color)
            im.save(tmp_path / cls / f"{i}.png")
    return tmp_path


def test_folder_pipeline_and_checkpoint_roundtrip(shapes_dir, tmp_path):
    d = get_datasets("folder", SIZE, data_dir=str(shapes_dir))
    assert d["classes"] == ["blue", "green", "red"] and len(d["train"]) > len(d["val"]) > 0
    x, y = d["train"][0]
    assert x.shape == (3, SIZE, SIZE) and 0 <= y < 3
    model = build_model("resnet18", 3, pretrained=False)
    save_checkpoint(tmp_path / "m.pt", model, d["classes"], "resnet18", SIZE)
    b = load_checkpoint(tmp_path / "m.pt", torch.device("cpu"))
    img, _ = d["raw_test"][0]
    out = explain(b, img)
    assert len(out["top"]) == 3 and out["overlay"].shape == (SIZE, SIZE, 3)
    assert abs(sum(p for _, p, _ in out["top"]) - 1) < 1e-4
