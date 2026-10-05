"""Noise- and PSF-aware multiband U-Net.

Encoder-decoder with skip connections over four scales. Each block is a residual pair of 3x3 convolutions with group
normalisation and swish activations. The PSF is not an image channel: a small encoder turns the per-band PSF stamps
into a 64-number embedding, which rescales and shifts every encoder feature map (FiLM, feature-wise linear
modulation), so the same weights adapt to any seeing. Heads read the last decoder layer (cfg["heads"] chooses which
of the optional ones a model has):

  galaxy_heatmap      probability of a galaxy centre at each pixel
  star_heatmap        probability of a star centre (optional)
  sfregion_map           probability that a pixel holds detectable light of a star-forming region (optional)
  tidal_map           probability that a pixel holds detectable light of a tidal stream or shell (optional)
  spike_map           probability that a pixel lies on a diffraction spike (optional)
  centroid_offset     sub-pixel offset from the peak pixel to the true centre
  source_structure    scaled log size, axis ratio, sin 2PA, cos 2PA
  detection_heatmap   probability of a source (galaxy or star) centre, from the decoder features and all the maps
                      above (optional): the output detections are taken from

Models made before the *_map heads had clump_heatmap and tidal_heatmap instead: centres of star-forming clumps and
tidal blobs. They can still be built (cfg["heads"] listing them) so their weights load.

The detection head starts as an exact copy of the galaxy heatmap (its learned correction starts at zero), so adding
it to a trained model changes nothing until it is trained; it then learns to add stars and to reject peaks that the
other maps say are star-forming regions, tidal features or spikes.

This is the same network as in lsst_unet_training (layer names and creation order included), so weights trained there
load unchanged. Only the inference parts are here; the losses and training live in lsst_unet_training.
"""

import keras
import numpy as np
from keras import layers, models

from .config import BANDS

# Heads that are maps of the image and feed the detection head, in the order they enter it.
MAP_HEADS = ("galaxy_heatmap", "star_heatmap", "sfregion_map", "tidal_map", "spike_map", "clump_heatmap",
             "tidal_heatmap")

HEATMAP_BIAS = -4.595  # sigmoid(-4.595) = 0.01: heatmaps start near "no source" everywhere


def normalisation_layer(filters, name):
    """Group normalisation with up to 8 groups that divide the filter count."""
    groups = min(8, filters)
    while filters % groups:
        groups -= 1
    return layers.GroupNormalization(groups=groups, axis=-1, name=name)


def conv_block(inputs, filters, name, dilation=1):
    """Residual block: conv-norm-swish-conv-norm, plus a (1x1-projected) shortcut, then swish."""
    x = layers.Conv2D(filters, 3, padding="same", dilation_rate=dilation, use_bias=False,
                      kernel_initializer="he_normal", name=f"{name}_conv1")(inputs)
    x = normalisation_layer(filters, f"{name}_norm1")(x)
    x = layers.Activation("swish", name=f"{name}_act1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal",
                      name=f"{name}_conv2")(x)
    x = normalisation_layer(filters, f"{name}_norm2")(x)
    shortcut = inputs
    if inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_shortcut")(shortcut)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("swish", name=f"{name}_out")(x)


def film(features, psf_embedding, filters, name):
    """FiLM: features * (1 + gamma) + beta, with gamma and beta predicted from the PSF embedding."""
    gamma = layers.Dense(filters, name=f"{name}_gamma")(psf_embedding)
    gamma = layers.Reshape((1, 1, filters), name=f"{name}_gamma_r")(gamma)
    beta = layers.Dense(filters, name=f"{name}_beta")(psf_embedding)
    beta = layers.Reshape((1, 1, filters), name=f"{name}_beta_r")(beta)
    one_plus_gamma = layers.Lambda(lambda t: 1.0 + t, name=f"{name}_gp1")(gamma)
    return layers.Add(name=f"{name}_film")([layers.Multiply(name=f"{name}_scale")([features, one_plus_gamma]), beta])


def psf_encoder(psf_stamps):
    """Per-band PSF stamps -> 64-number embedding."""
    x = layers.Conv2D(16, 3, padding="same", activation="swish", name="psf_c1")(psf_stamps)
    x = layers.Conv2D(32, 3, padding="same", activation="swish", name="psf_c2")(x)
    x = layers.GlobalAveragePooling2D(name="psf_gap")(x)
    return layers.Dense(64, activation="swish", name="psf_embed")(x)


def decoder_block(inputs, skip, filters, name):
    """Upsample x2, project, concatenate the encoder skip, then a conv block."""
    x = layers.UpSampling2D(size=2, interpolation="bilinear", name=f"{name}_up")(inputs)
    x = layers.Conv2D(filters, 1, padding="same", name=f"{name}_project")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    return conv_block(x, filters, name=f"{name}_conv")


def heatmap_head(features, base_filters, name):
    x = layers.Conv2D(base_filters, 3, padding="same", activation="swish", name=f"{name}_pre")(features)
    bias = keras.initializers.Constant(HEATMAP_BIAS)
    return layers.Conv2D(1, 1, activation="sigmoid", name=name, bias_initializer=bias)(x)  # pyright: ignore


def probability_logit(p):
    """log(p / (1 - p)) of a probability map, clipped away from 0 and 1."""
    p = keras.ops.clip(p, 1e-6, 1.0 - 1e-6)
    return keras.ops.log(p) - keras.ops.log(1.0 - p)


def detection_head(features, maps, base_filters):
    """Source-centre map from the decoder features and the other heads' maps (see the module docstring).

    logit = direct(logits of the maps) + correction(features, logits of the maps). direct starts as the identity on
    the galaxy map and correction at zero, so the head starts equal to the galaxy heatmap.
    """
    names = [name for name in MAP_HEADS if name in maps]
    logits = [layers.Lambda(probability_logit, name=f"detection_logit_{name}")(maps[name]) for name in names]
    stacked = layers.Concatenate(name="detection_maps")(logits) if len(logits) > 1 else logits[0]
    identity = np.zeros((1, 1, len(names), 1), np.float32)
    identity[0, 0, names.index("galaxy_heatmap"), 0] = 1.0
    direct = layers.Conv2D(1, 1, name="detection_direct", kernel_initializer=keras.initializers.Constant(identity),
                           bias_initializer="zeros")(stacked)
    context = layers.Concatenate(name="detection_context")([features, stacked])
    hidden = layers.Conv2D(base_filters, 3, padding="same", activation="swish", name="detection_pre")(context)
    correction = layers.Conv2D(1, 1, name="detection_correction", kernel_initializer="zeros",
                               bias_initializer="zeros")(hidden)
    logit = layers.Add(name="detection_sum")([direct, correction])
    return layers.Activation("sigmoid", name="detection_heatmap")(logit)


def build_unet(cfg):
    """The detector as a Keras model with inputs image_planes and psf_kernels and one output per head (cfg["heads"],
    plus centroid_offset and source_structure)."""
    filters, input_size = cfg["base_filters"], cfg["tile_size"] + 2 * cfg["tile_halo"]
    image_in = layers.Input((input_size, input_size, 2 * len(BANDS)), name="image_planes")
    psf_in = layers.Input((cfg["psf_stamp"], cfg["psf_stamp"], len(BANDS)), name="psf_kernels")
    psf_embedding = psf_encoder(psf_in)

    enc1 = film(conv_block(image_in, filters, "enc1"), psf_embedding, filters, "enc1_film")
    enc2 = film(conv_block(layers.MaxPool2D()(enc1), filters * 2, "enc2"), psf_embedding, filters * 2, "enc2_film")
    enc3 = film(conv_block(layers.MaxPool2D()(enc2), filters * 4, "enc3"), psf_embedding, filters * 4, "enc3_film")
    enc4 = film(conv_block(layers.MaxPool2D()(enc3), filters * 8, "enc4"), psf_embedding, filters * 8, "enc4_film")
    bottleneck = conv_block(layers.MaxPool2D()(enc4), filters * 12, "bottleneck", dilation=2)
    bottleneck = layers.SpatialDropout2D(0.15)(film(bottleneck, psf_embedding, filters * 12, "bottleneck_film"))

    dec4 = decoder_block(bottleneck, enc4, filters * 8, "dec4")
    dec3 = decoder_block(dec4, enc3, filters * 4, "dec3")
    dec2 = decoder_block(dec3, enc2, filters * 2, "dec2")
    dec1 = decoder_block(dec2, enc1, filters, "dec1")

    heads = set(cfg["heads"])
    outputs = {  # the original heads, in their original order
        "galaxy_heatmap": heatmap_head(dec1, filters, "galaxy_heatmap"),
        "centroid_offset": layers.Conv2D(2, 1, activation="tanh", name="centroid_offset")(dec1),
        "source_structure": layers.Conv2D(4, 1, activation=None, name="source_structure")(dec1),
    }
    for name in ("clump_heatmap", "tidal_heatmap", "star_heatmap", "sfregion_map", "tidal_map", "spike_map"):
        if name in heads:
            outputs[name] = heatmap_head(dec1, filters, name)
    if "detection_heatmap" in heads:
        outputs["detection_heatmap"] = detection_head(dec1, outputs, filters)
    return models.Model(inputs={"image_planes": image_in, "psf_kernels": psf_in}, outputs=outputs,
                        name="mep_multiband_unet")
