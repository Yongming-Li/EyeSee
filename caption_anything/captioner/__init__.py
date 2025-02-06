from .blip import BLIPCaptioner
from .blip2 import BLIP2Captioner
from .git import GITCaptioner
from .base_captioner import BaseCaptioner


def build_captioner(type, device, args=None):
    if type == 'blip2':
        return BLIPCaptioner(device, enable_filter=args.clip_filter)
    elif type == 'blip':
        return BLIP2Captioner(device, enable_filter=args.clip_filter)
    elif type == 'git':
        return GITCaptioner(device, enable_filter=args.clip_filter)
    else:
        raise NotImplementedError("")