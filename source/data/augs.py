import torch
from torchvision import transforms


def gauss_noise_tensor(sigma=0.1):
    def fn(img):
        out = img + sigma * torch.randn_like(img)
        out = torch.clamp(out, 0, 1)  #  pixel space is [0, 1]
        return out

    return fn


def augmentation_strong(noise=0.0, imsize=32):
    transform_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(imsize, scale=(0.2, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.AugMix(),
            transforms.ToTensor(),
            gauss_noise_tensor(noise) if noise > 0 else lambda x: x,
        ]
    )
    return transform_aug


def simclr_augmentation(imsize, hflip=False):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(imsize),
            transforms.RandomHorizontalFlip(0.5) if hflip else lambda x: x,
            get_color_distortion(s=0.5),
            transforms.ToTensor(),
        ]
    )


def random_Linf_noise(trnsfms: transforms.Compose = None, epsilon=64 / 255):
    if trnsfms is None:
        trnsfms = transforms.Compose([transforms.ToTensor()])

    randeps = torch.rand(1).item() * epsilon

    def fn(x):
        x = x + randeps * torch.randn_like(x).sign()
        return torch.clamp(x, 0, 1)

    trnsfms.transforms.append(fn)
    return trnsfms


def get_color_distortion(s=0.5):
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
