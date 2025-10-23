from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import SimpleITK
import numpy as np
import torch
import torchio as tio
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

from util.torchio_transform import HUToRSPTransform


class RSPImage(ABC):
    """
    Base class for RSP images.

    An RSP image is the product of a pCT scan, or a known phantom in a simulation.
    The RSP image can be rotated and translated arbitrarily in order to implement irradiation from any angle.
    """

    # Now, THIS... is revolutionary.
    RSP_WATER = 1.

    # Integrated over PSTAR data between 10 and 330 MeV with spline interpolation.
    # Rounded to 4 significant digits, since that is the PSTAR precision.
    RSP_AIR = 0.001064

    # Values from the paper defining the head phantom for pCT (mean experimental values):
    # Giacometti et al (2017), Development of a high resolution voxelised head phantom for medical physics applications
    RSP_SOFT_TISSUE = 1.032
    RSP_BRAIN_TISSUE = 1.044
    RSP_SPINAL_DISC = 1.069
    RSP_TRABECULAR_BONE = 1.111
    RSP_CORTICAL_BONE = 1.331
    RSP_TOOTH_DENTINE = 1.524
    RSP_TOOTH_ENAMEL = 1.651

    # Used to transform Hounsfield units into RSP values.
    # Taken from the AttenuationRange.dat file used in the simulation of the head_phantom.
    ATTENUATION_RANGES = [
        (-1025, -200,  RSP_AIR),
        (-200,  40,    RSP_SOFT_TISSUE),
        (40,    80,    RSP_BRAIN_TISSUE),
        (80,    190,   RSP_SPINAL_DISC),
        (190,   600,   RSP_TRABECULAR_BONE),
        (600,   1100,  RSP_CORTICAL_BONE),
        (1100,  1800,  RSP_TOOTH_DENTINE),
        (1800,  3100,  RSP_TOOTH_ENAMEL)
    ]

    def __init__(self, volume: Optional[np.ndarray] = np.array([272., 168., 200.]),
                 offset: Optional[np.ndarray] = np.array([0., 0., 0]),
                 voxel_size: float = 1.,
                 rotation_angle: float = 0.,
                 rotation_axis: Optional[np.ndarray] = np.array([0., 1., 0.]),
                 align_to_y: bool = False,
                 flip_x_y: bool = False,
                 padding_value: float = RSP_AIR):
        """
        Initializes general properties of the RSP image such as offset and rotation as well as the size of the
        voxels.

        Voxels can only be cubes in order to enable rotation through affine transformation. Hence, the voxel size
        is given as a scalar value.

        The volume represented by the RSP image can be any size. It does not specify the scale of the model, but
        only the viewport, i.e. it is possible to crop the given RSP image by specifying a smaller represented volume.
        The image can also be padded with air, if the given image does not fill the entire space.
        The volume should be consistent throughout the application to be able to make RSP images compatible.

        If a rotation and a translation are specified, the rotation is applied first before translating the rotated
        phantom.

        :param volume: The size of the represented volume as an np.ndarray with 3 values for x, y and z size. The
            volume should be divisible by the given voxel size. The sizes usually represent mm.
            Optional, default: (272, 168, 200)
        :param offset: The translation of the phantom from the origin as an np.ndarray with 3 values (x, y, z).
            Optional, default: (0, 0, 0)
        :param voxel_size: The side length of the voxel cubes. Optional, default: 1
        :param rotation_angle: The angle in degrees by which to rotate the phantom around rotation_axis. Optional,
            default: 0
        :param rotation_axis: The unit axis around which to rotate the phantom. Optional, default: (0, 1, 0)
        :param align_to_y: Should the image be changed from the z-axis pointing to the top to the y-axis pointing to
            the top. Default: False
        :param flip_x_y: Flip x and y-axis. Can be useful for certain data, such as DoseActor. Optional, default: False
        :param padding_value: The value to assume being outside of the image's volume. Optional, default: RSP_AIR
        """

        self.volume = volume
        self.offset = offset
        self.voxel_size = voxel_size
        self.voxel_count = tuple((self.volume / self.voxel_size).astype(int))
        self.rotation_angle = rotation_angle
        self.rotation_axis = rotation_axis
        self.padding_value = padding_value

        image = self._load_image()

        # Transform to RAS+ (right, anterior, superior)
        canon = tio.ToCanonical()
        image = canon(image)

        # If the spacing is not correct yet, resample
        if image.spacing != (voxel_size, voxel_size, voxel_size):
            resample = tio.Resample(voxel_size)
            image = resample(image)

        if align_to_y:
            ct = image.numpy()
            ct = np.swapaxes(ct, 2, 3).copy()
            image.set_data(ct)

        if flip_x_y:
            ct = image.numpy()
            ct = np.flip(ct, axis=1)
            ct = np.flip(ct, axis=2)
            image.set_data(ct.copy())

        # If the viewport is not correct yet, crop or pad
        if image.shape != (1,) + self.voxel_count:
            resize = tio.CropOrPad(self.voxel_count, padding_mode=self.padding_value)
            image = resize(image)

        self.image = image

    @abstractmethod
    def _load_image(self) -> tio.Image:
        """
        This method has to be overridden in subclasses to provide the actual RSP image.

        :return: The tio.Image representing the RSP values in the represented phantom.
        """
        return tio.Image()

    def get_world_voxels(self):
        """
        Transforms the image represented by the RSPImage object according to the defined rotation axis and angle.

        :return: A numpy array containing the rotated voxels' RSP values. The dimensions are as defined through the
            constructor, i.e. volume / voxel_size
        """
        ct = np.squeeze(self.image.numpy())
        rotation = Rotation.from_rotvec(np.radians(self.rotation_angle) * self.rotation_axis).as_matrix()
        center = 0.5 * np.array(ct.shape) - 0.5
        offset = center - (center+self.offset).dot(rotation)
        ct = affine_transform(ct, rotation.T, offset=offset, cval=self.padding_value)
        return ct


class WaterCuboidRSPImage(RSPImage):
    """
    Represents a cuboid made of water with specified dimensions.
    """

    def __init__(self, phantom_size: np.ndarray, **kwargs):
        """
        Initializes a water cuboid with the specified dimensions.

        :param phantom_size: Dimensions of the water cuboid
        :param kwargs: Optional keyword arguments for the superclass.
        """
        self.phantom_size = phantom_size
        super(WaterCuboidRSPImage, self).__init__(**kwargs)

    def _load_image(self) -> tio.Image:
        """
        Generates a water cuboid of the size specified in the constructor.

        Voxel size is assumed to be 1.

        :return: A ScalarImage with the RSP of water (1.0) and the specified size.
        """
        cuboid = torch.mul(torch.ones((1,) + tuple(self.phantom_size)), self.RSP_WATER)
        return tio.ScalarImage(tensor=cuboid)


class MetaImageRSPImage(RSPImage):
    """
    Loads a MetaImage file through SimpleITK and turns it into a TorchIO Image.

    The image can optionally be cropped in case there is a border of dummy values around the actual image.
    """

    def __init__(self, image_src: Union[str, SimpleITK.Image],
                 crop_by: Optional[Union[int, Tuple[int, int, int], Tuple[int, int, int, int, int, int]]] = 1,
                 transform_hu_to_rsp: bool = True,
                 align_to_y: bool = True,
                 **kwargs):
        """
        Initializes an RSPImage based on a MetaImage file.

        The image can optionally be cropped in case there is a border of dummy values around the actual image.
        The cropping is specified in the same fashion as the TorchIO Crop transform accepts it, i.e. a single value
        will crop by that much on all sides, 3 values crop by the specified amount on both sides of the individual
        axes, and 6 values specify the values to crop on each of the 6 sides of the image.

        :param image_src: The path and name of the MetaImage file or a SimpleITK.Image containing the image.
        :param crop_by: The cropping value either with 1 value for all sides, symmetric cropping with 1 value per axis,
            or 6 values representing each side. Optional, default: 1
        :param transform_hu_to_rsp: Should the values from the MetaImage be converted from HU to RSP? Optional,
            default: True
        :param align_to_y: Should the image be changed from the z-axis pointing to the top to the y-axis pointing to
            the top. This is passed to the superclass and is listed here to assign a new default value.
            Optional, default: True
        :param kwargs: Optional keyword arguments passed along to the superclass initializer.
        """
        self.image_src = image_src
        self.crop_by = crop_by
        self.transform_hu_to_rsp = transform_hu_to_rsp
        super(MetaImageRSPImage, self).__init__(align_to_y=align_to_y, **kwargs)

    def _load_image(self) -> tio.Image:
        """
        Loads an RSPImage based on the given MetaImage file.

        The loaded MetaImage is assumed to contain HU values and will be translated to RSP with the attenuation
        ranges defined in the superclass.

        The crop specified in the constructor is performed directly after loading the image.

        The MetaImage also defines the voxel size (may not be cubic), which will be left as is and transformed
        to the desired voxel size in the superclass.

        :return: A ScalarImage based on a MetaImage file.
        """
        itk_image: Optional[SimpleITK.Image] = None
        if isinstance(self.image_src, str):
            itk_image = SimpleITK.ReadImage(self.image_src)
        elif isinstance(self.image_src, SimpleITK.Image):
            itk_image = self.image_src

        if itk_image is None:
            raise ValueError("Invalid image source")

        image = tio.ScalarImage.from_sitk(itk_image)

        if self.crop_by is not None:
            crop = tio.Crop(self.crop_by)
            image = crop(image)

        if self.transform_hu_to_rsp:
            hu_to_rsp = HUToRSPTransform(self.ATTENUATION_RANGES)
            image = hu_to_rsp(image)

        return image
