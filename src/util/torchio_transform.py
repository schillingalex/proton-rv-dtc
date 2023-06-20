import torch
import torchio as tio


class HUToRSPTransform(tio.IntensityTransform):
    """
    TorchIO intensity transform that converts an image from Hounsfield units to relative stopping power values.

    The transform needs a list of tuples defining the lower bound, upper bound, and RSP value to translate to.
    Example: [(-1010, -200,  0.001)]
    """

    def __init__(self, attenuation_ranges, **kwargs):
        """
        Initializes with a list of attenuation ranges.

        :param attenuation_ranges: A list of tuples containing lower bound, upper bound, and RSP value to translate to.
        :param kwargs: Arbitrary arguments for the base class.
        """
        super().__init__(**kwargs)
        self.attenuation_ranges = attenuation_ranges
        self.args_names = ("attenuation_ranges",)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Translates the HU in the input to the RSP defined in the attenuation ranges.

        :param subject: The subject to transform.
        :return: The subject with RSP values instead of HU.
        """
        for image in self.get_images(subject):
            old_data = image.data
            new_data = torch.zeros_like(old_data)
            for min_hu, max_hu, rsp in self.attenuation_ranges:
                mask = (old_data > min_hu) & (old_data <= max_hu)
                new_data += mask * rsp
            image.set_data(new_data)
        return subject
