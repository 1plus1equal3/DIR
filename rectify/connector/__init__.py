from .identity import Indentity
from .pooling import MaxPooling, AveragePooling
from .upsampling import Upsampling
from .conv import DownConv, UpConv
from .shuffle import PixelShuffle, PixelUnshuffle

CONNECTOR = {
    "identity": Indentity,
    "max_pooling": MaxPooling,
    "avg_pooling": AveragePooling,
    "upsampling": Upsampling,
    "down_conv": DownConv,
    "up_conv": UpConv,
    "pixel_shuffle": PixelShuffle,
    "pixel_unshuffle": PixelUnshuffle,
}