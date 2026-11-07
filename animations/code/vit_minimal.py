"""Code for the lecture-4 slide "ViT in PyTorch": the Vision Transformer (ViT-B/16) in a dozen lines, checked against
Google's released model.

The class below is the code on the slide. To show that it is the whole model, we copy the weights of
google/vit-base-patch16-224 (pre-trained on ImageNet-21k, fine-tuned on ImageNet-1k at 224 x 224) into it and compare
its class scores with those of the Hugging Face implementation on real ImageNet validation images (the Imagenette
subset: 10 classes, 13 images each). Also checked: the parameter count (86.6M, the "86M" of the paper's Table 1) and
that the stride-16 convolution is exactly "flatten each 16 x 16 x 3 patch, multiply by a 768 x D matrix".
Writes data/vit_minimal.json.

Run:  python vit_minimal.py      (needs data/imagenette2-320/val, see vit_inspect.py)
Try:  P=32 (ViT-B/32): 7 x 7 = 49 tokens instead of 196, about 4x cheaper attention.
"""
import json
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent
OUT = HERE / "data" / "vit_minimal.json"
VAL = HERE / "data" / "imagenette2-320" / "val"


# ---------------------------------------------------------------- the code on the slide
class ViT(nn.Module):
    def __init__(self, P=16, D=768, L=12, H=12, n_cls=1000, N=196):
        super().__init__()
        self.patch = nn.Conv2d(3, D, kernel_size=P, stride=P)  # = flatten each P x P x 3 patch, multiply by E
        self.cls = nn.Parameter(torch.zeros(1, 1, D))          # the [CLS] token, as in BERT
        self.pos = nn.Parameter(torch.zeros(1, N + 1, D))      # one learned position vector per token
        block = nn.TransformerEncoderLayer(D, H, 4 * D, dropout=0.0, activation="gelu",
                    layer_norm_eps=1e-12, batch_first=True, norm_first=True)   # pre-norm block
        self.blocks = nn.TransformerEncoder(block, L, enable_nested_tensor=False)
        self.norm, self.head = nn.LayerNorm(D, eps=1e-12), nn.Linear(D, n_cls)

    def forward(self, img):                                    # B x 3 x 224 x 224
        x = self.patch(img).flatten(2).transpose(1, 2)         # B x 196 x 768
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos   # B x 197 x 768
        x = self.blocks(x)                                     # no mask: every patch reads every patch
        return self.head(self.norm(x[:, 0]))                   # classify from the [CLS] output


# ---------------------------------------------------------------- loading Google's weights into it
def load_google(vit, hf):
    sd = hf.state_dict()
    g = lambda k: sd[k]
    vit.patch.weight.data.copy_(g("vit.embeddings.patch_embeddings.projection.weight"))
    vit.patch.bias.data.copy_(g("vit.embeddings.patch_embeddings.projection.bias"))
    vit.cls.data.copy_(g("vit.embeddings.cls_token"))
    vit.pos.data.copy_(g("vit.embeddings.position_embeddings"))
    for i, blk in enumerate(vit.blocks.layers):
        p = f"vit.layers.{i}."
        att = blk.self_attn
        att.in_proj_weight.data.copy_(torch.cat([g(p + f"attention.{n}_proj.weight") for n in "qkv"]))
        att.in_proj_bias.data.copy_(torch.cat([g(p + f"attention.{n}_proj.bias") for n in "qkv"]))
        att.out_proj.weight.data.copy_(g(p + "attention.o_proj.weight"))
        att.out_proj.bias.data.copy_(g(p + "attention.o_proj.bias"))
        blk.linear1.weight.data.copy_(g(p + "mlp.fc1.weight")); blk.linear1.bias.data.copy_(g(p + "mlp.fc1.bias"))
        blk.linear2.weight.data.copy_(g(p + "mlp.fc2.weight")); blk.linear2.bias.data.copy_(g(p + "mlp.fc2.bias"))
        blk.norm1.weight.data.copy_(g(p + "layernorm_before.weight")); blk.norm1.bias.data.copy_(g(p + "layernorm_before.bias"))
        blk.norm2.weight.data.copy_(g(p + "layernorm_after.weight")); blk.norm2.bias.data.copy_(g(p + "layernorm_after.bias"))
    vit.norm.weight.data.copy_(g("vit.layernorm.weight")); vit.norm.bias.data.copy_(g("vit.layernorm.bias"))
    vit.head.weight.data.copy_(g("classifier.weight")); vit.head.bias.data.copy_(g("classifier.bias"))
    return vit


# the Imagenette classes and their ImageNet-1k indices
CLASSES = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
           "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}


def images(per_class=13):
    from PIL import Image
    out = []
    for wnid, idx in CLASSES.items():
        for f in sorted((VAL / wnid).glob("*.JPEG"))[:per_class]:
            out.append((Image.open(f).convert("RGB"), idx, f.name))
    return out


if __name__ == "__main__":
    from transformers import ViTForImageClassification, ViTImageProcessor
    torch.manual_seed(0)
    name = "google/vit-base-patch16-224"
    hf = ViTForImageClassification.from_pretrained(name, attn_implementation="eager").eval()
    proc = ViTImageProcessor.from_pretrained(name)
    assert hf.config.layer_norm_eps == 1e-12 and hf.config.hidden_act == "gelu"
    vit = load_google(ViT(), hf).eval()
    n_params = sum(p.numel() for p in vit.parameters())

    # the patch embedding is a linear map of the flattened patch
    x = torch.randn(2, 3, 224, 224)
    conv = vit.patch(x).flatten(2).transpose(1, 2)                                  # 2 x 196 x 768
    patches = x.unfold(2, 16, 16).unfold(3, 16, 16)                                 # 2 x 3 x 14 x 14 x 16 x 16
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 196, 768)               # flatten each 3 x 16 x 16 patch
    lin = patches @ vit.patch.weight.reshape(768, -1).T + vit.patch.bias
    conv_err = (conv - lin).abs().max().item()

    data = images()
    batch = proc(images=[im for im, _, _ in data], return_tensors="pt")["pixel_values"]
    labels = torch.tensor([y for _, y, _ in data])
    with torch.no_grad():
        ours = torch.cat([vit(b) for b in batch.split(26)])
        ref = torch.cat([hf(pixel_values=b).logits for b in batch.split(26)])
    diff = (ours - ref).abs().max().item()
    agree = (ours.argmax(1) == ref.argmax(1)).float().mean().item()
    acc = (ours.argmax(1) == labels).float().mean().item()
    res = {"model": name, "params": n_params, "conv_vs_linear_max_abs_diff": conv_err,
           "images": len(data), "max_abs_logit_diff_vs_hf": diff, "top1_agreement_with_hf": agree,
           "top1_accuracy_ours": acc}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert n_params == 86_567_656 and conv_err < 1e-4 and diff < 1e-3 and agree == 1.0
    print("saved", OUT)
