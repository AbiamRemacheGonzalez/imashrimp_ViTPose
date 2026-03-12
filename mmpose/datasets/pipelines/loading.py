# Copyright (c) OpenMMLab. All rights reserved.
import imashrimp_mmcv.mmcv as mmcv
import numpy as np
import os

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_image(self, path):
        img_bytes = self.file_client.get(path)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if img is None:
            raise ValueError(f'Fail to read {path}')
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def __call__(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        image_file = results.get('image_file', None)

        if isinstance(image_file, (list, tuple)):
            # Load images from a list of paths
            results['img'] = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            # Load single image from path
            results['img'] = self._read_image(image_file)
        else:
            if 'img' not in results:
                # If `image_file`` is not in results, check the `img` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError('Either `image_file` or `img` should exist in '
                               'results.')
            assert isinstance(results['img'], np.ndarray)
            if self.color_type == 'color' and self.channel_order == 'rgb':
                # The original results['img'] is assumed to be image(s) in BGR
                # order, so we convert the color according to the arguments.
                if results['img'].ndim == 3:
                    results['img'] = mmcv.bgr2rgb(results['img'])
                elif results['img'].ndim == 4:
                    results['img'] = np.concatenate(
                        [mmcv.bgr2rgb(img) for img in results['img']], axis=0)
                else:
                    raise ValueError('results["img"] has invalid shape '
                                     f'{results["img"].shape}')

            results['image_file'] = None

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImashrimpImageFromFile:
    """
    Loading RGBD image(s) from file.

    It also allows RGB images to be read if depth_file does not exist.

    It also adds an indirect image and depth channel reading system. The latter method was used to save space.
    """
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_image(self, path):
        try:
            i_path = self._read_line(path)
            if os.path.exists(i_path):
                path = i_path
            img_bytes = self.file_client.get(path)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, channel_order=self.channel_order)
            if img is None:
                raise ValueError(f'Fail to read {path}')
            if self.to_float32:
                img = img.astype(np.float32)
            return img
        except IOError as e:
            raise ValueError(f'Fail to read {path}: {e}')
            return None

    def _read_depth_file(self, path):
        try:
            d_path = self._read_line(path)
            if os.path.exists(d_path):
                return np.load(d_path)
            else:
                return np.load(path)
        except IOError as e:
            raise ValueError(f'Fail to read {path}: {e}')
            return None

    @staticmethod
    def _read_line(path):
        with open(path, 'r') as f:
            try:
                linea = f.readline()
            except Exception as e:
                return ""
        return linea

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
            self.file_client = mmcv.FileClient(**self.file_client_args)
        image_file = results.get('image_file', None)
        depth_file = results.get('depth_file', None)

        # Tratamiento del canal de profundidad se normaliza para que esté entre 0 y 255, manteniendo la proporción de los valores originales
        if depth_file:
            depth_image = self._read_depth_file(depth_file)
            depth_image = np.expand_dims(depth_image, axis=2)
            datos_no_cero = depth_image[depth_image != 0.]
            min_val = datos_no_cero.min()
            max_val = depth_image.max()
            depth_image_nor = ((depth_image - min_val) / (max_val - min_val)) * 255
            depth_image_nor[depth_image_nor < 0] = 0

        if isinstance(image_file, (list, tuple)):
            # Load images from a list of paths
            rgb_image = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            # Load single image from path
            rgb_image = self._read_image(image_file)
        else:
            if 'img' not in results:
                # If `image_file`` is not in results, check the `img` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError('Either `image_file` or `img` should exist in '
                               'results.')
            assert isinstance(results['img'], np.ndarray)
            if self.color_type == 'color' and self.channel_order == 'rgb':
                # The original results['img'] is assumed to be image(s) in BGR
                # order, so we convert the color according to the arguments.
                if results['img'].ndim == 3:
                    results['img'] = mmcv.bgr2rgb(results['img'])
                elif results['img'].ndim == 4:
                    results['img'] = np.concatenate(
                        [mmcv.bgr2rgb(img) for img in results['img']], axis=0)
                    raise KeyError('Uncontrolled error. In principle, it should never appear here. mmpose/datasets/pipelines/loading_depth/__call__')
                else:
                    raise ValueError('results["img"] has invalid shape '
                                     f'{results["img"].shape}')
            results['image_file'] = None

        if depth_file:
            image_rgbd = np.concatenate((rgb_image, depth_image_nor), axis=2)
            image_rgbd = image_rgbd.astype(np.float32)
            results['img'] = image_rgbd
        else:
            results['img'] = rgb_image
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
