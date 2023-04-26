# DSDL Detection Dataset

## 1. Abstract

Data is the cornerstone of artificial intelligence. The efficiency of data acquisition, exchange, and application directly impacts the advances in technologies and applications. Over the long history of AI, a vast quantity of data sets have been developed and distributed. However, these datasets are defined in very different forms, which incurs significant overhead when it comes to exchange, integration, and utilization -- it is often the case that one needs to develop a new customized tool or script in order to incorporate a new dataset into a workflow.

To overcome such difficulties, we develop **Data Set Description Language (DSDL)**. More details please visit our [official documents](https://opendatalab.github.io/dsdl-docs/getting_started/overview/), dsdl datasets can be downloaded from our platform [OpenDataLab](https://opendatalab.com/).

## 2. Steps

- install dsdl:

  install by pip:

  ```
  pip install dsdl
  ```

  install by source code:

  ```
  git clone https://github.com/opendatalab/dsdl-sdk.git -b schema-dsdl
  cd dsdl-sdk
  python setup.py install
  ```

- install mmdet and pytorch:
  please refer this [installation documents](https://mmdetection.readthedocs.io/en/3.x/get_started.html).

- train:

  - using single gpu:

  ```
  python tools/train.py {config_file}
  ```

  - using slrum:

  ```
  ./tools/slurm_train.sh {partition} {job_name} {config_file} {work_dir} {gpu_nums}
  ```

## 3. Test Results

  | Datasets |                                                                    Model                                                                     | mAP |            Config            |
  | :------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :----: | :-----: | :--------------------------: |
  |   DIOR   | [model](https://download.openmmlab.com/mmrotate/v1.0/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dior/rotated-retinanet-rbox-le90_r50_fpn_1x_dior-caf9143c.pth) |  53.95  | [config](./dior.py) |
  |   DOTAV1   | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth) |  53.95  | [config](./dotav1.py) |