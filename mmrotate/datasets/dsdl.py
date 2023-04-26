# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List

from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS
from dsdl.geometry import RBBox

try:
    from dsdl.dataset import DSDLDataset
except ImportError:
    DSDLDataset = None


@DATASETS.register_module()
class DSDLRotDataset(BaseDataset):
    """Dataset for dsdl detection.

    Args:
        specific_key_path(dict): Path of specific key which can not
            be loaded by it's field name.
        pre_transform(dict): pre-transform functions before loading.
    """

    METAINFO = {}

    def __init__(self,
                 specific_key_path: dict = {},
                 pre_transform: dict = {},
                 **kwargs) -> None:

        if DSDLDataset is None:
            raise RuntimeError(
                'Package dsdl is not installed. Please run "pip install dsdl".'
            )

        self.specific_key_path = specific_key_path

        loc_config = dict(type='LocalFileReader', working_dir='')
        if kwargs.get('data_root'):
            kwargs['ann_file'] = os.path.join(kwargs['data_root'],
                                              kwargs['ann_file'])
        self.required_fields = ['Image', 'ImageShape', 'Label', 'RotatedBBox', 'ignore_flag', ]

        self.extra_keys = [
            key for key in self.specific_key_path.keys()
            if key not in self.required_fields
        ]

        self.dsdldataset = DSDLDataset(
            dsdl_yaml=kwargs['ann_file'],
            location_config=loc_config,
            required_fields=self.required_fields,
            specific_key_path=specific_key_path,
            transform=pre_transform,
        )

        BaseDataset.__init__(self, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data info from an dsdl yaml file named as ``self.ann_file``

        Returns:
            List[dict]: A list of data info.
        """
        old_classes = ['plane', 'baseball_diamond', 'bridge', 'ground_track_field',
         'small_vehicle', 'large_vehicle', 'ship', 'tennis_court',
         'basketball_court', 'storage_tank', 'soccer_ball_field', 'roundabout',
         'harbor', 'swimming_pool', 'helicopter']
        self._metainfo['classes'] = old_classes # tuple(self.dsdldataset.class_names)

        data_list = []

        for i, data in enumerate(self.dsdldataset):
            # basic image info, including image id, path and size.
            datainfo = dict(
                img_id=i,
                img_path=os.path.join(self.data_prefix['img_path'],
                                      data['Image'][0].location),
                width=data['ImageShape'][0].width,
                height=data['ImageShape'][0].height,
            )

            # load instance info
            instances = []
            if 'RotatedBBox' in data.keys():
                for idx in range(len(data['RotatedBBox'])):
                    instance = {}
                    
                    rbbox = data['RotatedBBox'][idx]
                    if rbbox._polygon:
                        new_poly_arr = []
                        for point in rbbox._polygon:
                            new_poly_arr.append(point[0])
                            new_poly_arr.append(point[1])
                        instance["bbox"] = new_poly_arr
                    elif rbbox._rbbox:
                        rbbox2polygon_arr = RBBox.rbbox2polygon(rbbox._rbbox)
                        new_poly_arr1 = []
                        for point1 in rbbox2polygon_arr:
                            new_poly_arr1.append(point1[0])
                            new_poly_arr1.append(point1[1])
                        instance["bbox"] = new_poly_arr1
                    
                    if old_classes.index(data['Label'][idx].category_name) == 1:
                        
                        instance['bbox_label'] = old_classes.index(data['Label'][idx].category_name)  #data['Label'][idx].index_in_domain() - 1
                    else:
                        instance['bbox_label'] = 0

                    if 'ignore_flag' in data.keys():
                        # get ignore flag
                        instance['ignore_flag'] = data['ignore_flag'][idx]
                    else:
                        instance['ignore_flag'] = 0

                    for key in self.extra_keys:
                        # load extra instance info
                        instance[key] = data[key][idx]

                    instances.append(instance)

            datainfo['instances'] = instances
            # append a standard sample in data list
            if len(datainfo['instances']) > 0:
                data_list.append(datainfo)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_list = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_list.append(data_info)

        return valid_data_list
