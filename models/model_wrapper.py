from models.model import av_Mossformer_ConvTasnet
from models.dprnn import DPRNN
from models.mossformer import MossFormer2_SS
from models.av_tfgridnetV3.av_tfgridnetv3_separator import av_TFGridNetV3
from models.TFNet import TFNetSeparator
from models.TFNet_causal import CausalTFNetSeparator
from models.SpatialNet import SpatialNetSeparator
from models.OnlineSpatialNet import OnlineSpatialNetSeparator
def get_model(args):
    if args.network_audio['backbone'] == "convtasnet" or args.network_audio['backbone'] == "convtasnet_loud":
        model = av_Mossformer_ConvTasnet(args)

    if args.network_audio['backbone'] == "dprnn":
        model = DPRNN(args)
    
    if args.network_audio['backbone'] == "mossformer":
        model = MossFormer2_SS(args)

    if args.network_audio['backbone'] == "tfgridnet":
        model = av_TFGridNetV3(args)

    if args.network_audio['backbone'] == "tfnet":
        model = TFNetSeparator(args)

    if args.network_audio['backbone'] == "tfnet_causal":
        model = CausalTFNetSeparator(args)

    if args.network_audio['backbone'] == "spatialnet":
        model = SpatialNetSeparator(args)

    if args.network_audio['backbone'] == "online_spatialnet":
        model = OnlineSpatialNetSeparator(args)

    return model
