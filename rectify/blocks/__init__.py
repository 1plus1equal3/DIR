from .convolution import *

BLOCKS = {
    'conv': SimpleConvBlock,
    'res_conv': ResidualConvBlock,
    'cbam': CBAM,
    'dconv': DConvBlock,
    'dynamic_conv': Dynamic_Conv2D,
    'inception_block': InceptionBlock,
    'inception_block_v2': InceptionBlock_v2,
    'channel_se': ChannelSELayer,
    'spatial_se': SpatialSELayer,
    'channel_spatial_se': ChannelSpatialSELayer,
}
