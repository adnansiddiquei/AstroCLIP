import numpy as np


class ToRGB:
    """
    Transformation from raw image data (g, r, z) to the rgb values displayed at the legacy
    viewer https://www.legacysurvey.org/viewer

    Code copied from https://github.com/legacysurvey/imagine

    Code has been slightly edited to handle different shaped inputs, but is mainly identical.
    """

    def __init__(self, scales=None, m=0.03, Q=20, bands=['g', 'r', 'z']):
        rgb_scales = {
            'u': (2, 1.5),
            'g': (2, 6.0),
            'r': (1, 3.4),
            'i': (0, 1.0),
            'z': (0, 2.2),
        }
        if scales is not None:
            rgb_scales.update(scales)

        self.rgb_scales = rgb_scales
        self.m = m
        self.Q = Q
        self.bands = bands
        self.axes, self.scales = zip(*[rgb_scales[bands[i]] for i in range(len(bands))])

        # rearange scales to correspond to image channels after swapping
        self.scales = [self.scales[i] for i in self.axes]

    def __call__(self, imgs):
        # the below two lines have been added to handle a batch of images
        if len(imgs.shape) == 4:
            return np.stack([self(imgs[i]) for i in range(imgs.shape[0])])

        # Check image shape and set to C x H x W
        if imgs.shape[0] != len(self.bands):
            imgs = np.transpose(imgs, (2, 0, 1))

        I = 0
        for img, band in zip(imgs, self.bands):
            plane, scale = self.rgb_scales[band]
            img = np.maximum(0, img * scale + self.m)
            I = I + img
        I /= len(self.bands)

        Q = 20
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.0) * 1e-6
        H, W = I.shape
        rgb = np.zeros((H, W, 3), np.float32)
        for img, band in zip(imgs, self.bands):
            plane, scale = self.rgb_scales[band]
            rgb[:, :, plane] = (img * scale + self.m) * fI / I

        rgb = np.clip(rgb, 0, 1)

        return rgb
