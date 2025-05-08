'''
Custom model initiation using segmentation_models_pytorch

'''

import torch

# for predefined model architectures
import segmentation_models_pytorch as smp


def initialize_model(model_arch, GPU_lst, *args):
    if model_arch == 'smpUNet':
        model = initialize_smpUNet(*args)
    elif model_arch == 'smpUNetPP':
        model = initialize_smpUNetPP(*args)
    elif model_arch == 'smpMAnet':
        model = initialize_smpMAnet(*args)
    elif model_arch == 'smpDeepLabV3':
        model = initialize_DeepLabV3(*args)
    elif model_arch == 'smpDeepLabV3plus':
        model = initialize_DeepLabV3plus(*args)


    if torch.cuda.is_available() and GPU_lst is not None and GPU_lst[0] != 99:
        model = torch.nn.DataParallel(model, device_ids=GPU_lst)
        # to check GPUs could use: torch.cuda.device_count()
    return model


def initialize_smpUNet(
        n_channels, n_classes, device, use_batchnorm, outSize):
    '''
    https://smp.readthedocs.io/en/latest/models.html#id2
    '''
    unet = smp.Unet(
        encoder_name="resnet34", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None, # if not none use pre-trained weights e.g. 'imagenet' for encoder initialization
        decoder_use_batchnorm=use_batchnorm,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        in_channels=n_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=n_classes).to(device)  # model output channels (number of classes in dataset)
    return unet


def initialize_smpUNetPP(
        n_channels, n_classes, device, use_batchnorm, outSize):
    '''
    https://smp.readthedocs.io/en/latest/models.html#id2
    '''
    unet = smp.UnetPlusPlus(
        encoder_name='resnet34', encoder_depth=5,
        encoder_weights=None,
        decoder_use_batchnorm=use_batchnorm,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type=None, in_channels=n_channels, classes=n_classes,
        activation=None,  # An activation function to apply after the final convolution layer
        aux_params=None).to(device)
    return unet

def initialize_smpMAnet(
        n_channels, n_classes, device, use_batchnorm, outSize):
    '''
    https://smp.readthedocs.io/en/latest/models.html#id2
    '''
    manet = smp.MAnet(
        encoder_name='resnet34', encoder_depth=5, encoder_weights=None,  # 'imagenet',
        decoder_use_batchnorm=use_batchnorm,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_pab_channels=64, in_channels=n_channels, classes=n_classes,
        activation=None, aux_params=None).to(device)
    return manet


def initialize_DeepLabV3(
        n_channels, n_classes, device, use_batchnorm, outSize):
    '''
    https://smp.readthedocs.io/en/latest/models.html#id2
    '''
    manet = smp.DeepLabV3(
        encoder_name='resnet34', encoder_weights=None,
        decoder_channels=256, encoder_depth=5,
        in_channels=n_channels, classes=n_classes,
        activation=None).to(device)
    return manet


def initialize_DeepLabV3plus(
        n_channels, n_classes, device, use_batchnorm, outSize):
    '''
    https://smp.readthedocs.io/en/latest/models.html#id2
    '''
    manet = smp.DeepLabV3Plus(
        encoder_name='resnet34', encoder_weights=None,
        decoder_channels=256, encoder_depth=5,
        encoder_output_stride=16,
        decoder_atrous_rates=(12, 24, 36),
        in_channels=n_channels, classes=n_classes,
        activation=None).to(device)

    return manet

