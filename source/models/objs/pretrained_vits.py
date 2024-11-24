import torch
import torch.nn as nn
import timm
from timm.models import VisionTransformer
from source.layers.common_layers import RGBNormalize


class ViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.norm = RGBNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def forward(self, x):
        x = self.norm(x)
        return self.model(x)


def load_dino():
    model = timm.create_model(
        "vit_base_patch16_224_dino", pretrained=True, img_size=256
    )
    model = ViTWrapper(model).cuda()
    model.psize = 16
    return model


def load_dinov2(imsize=256):
    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m", pretrained=True, img_size=imsize
    )
    model = ViTWrapper(model).cuda()
    model.psize = 16
    return model


def load_mocov3():
    from timm.models.vision_transformer import vit_base_patch16_224

    model = vit_base_patch16_224(pretrained=False, dynamic_img_size=True)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
        map_location="cpu",
    )
    # Load the MoCo v3 state dict into the model
    state_dict = checkpoint["state_dict"]
    new_state_dict = {
        k.replace("module.base_encoder.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = ViTWrapper(model).cuda()
    model.img_size = 256
    model.psize = 16
    return model


def load_mae():
    from timm.models.vision_transformer import vit_base_patch16_224

    model = vit_base_patch16_224(pretrained=False, dynamic_img_size=True)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        map_location="cpu",
    )  # Load the state_dict into the model
    model.load_state_dict(
        checkpoint["model"], strict=False
    )  # Set the model to evaluation mode model.eval()
    model.eval()
    model = ViTWrapper(model).cuda()
    model.img_size = 256
    model.psize = 16
    return model
