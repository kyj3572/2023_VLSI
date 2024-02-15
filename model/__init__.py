from .rlfn import RLFN
from .rlfn_s import RLFN_S
from .abpn import ABPN
from .innopeak import InnoPeak

def get_model(args):
    model = args.model
    upscale = args.hr_size // args.lr_size
    feature = args.feature
    norm = args.normalization
    
    if model == 'rlfn':
        return RLFN(feature_channels=feature, upscale=upscale)
    
    if model == 'rlfn_s':
        return RLFN_S(feature_channels=feature, upscale=upscale)
    
    if model == 'abpn':
        return ABPN(mid_channels=feature, upscale=upscale, normalization=norm)
    
    if model == 'innopeak':
        return InnoPeak(upscale=upscale)