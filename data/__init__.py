'''create dataset and dataloader'''
import logging


def create_dataset_Fusion(dataset_opt, phase, opt):
    """Create a structure/function medical image fusion dataset."""
    fusion_task = opt['model']['fusion_task']

    # MRI/CT-like anatomical images are always the structure source.
    # PET/SPECT-like images are always the function source.
    structure_path = dataset_opt['dataroot_structure']
    function_path = dataset_opt['dataroot_function']

    fusion_mode = opt['model']['Fusion']['mode']
    if fusion_mode == 'MAX':
        from data.Fusion_dataset import FusionDataset
        dataset = FusionDataset(
            split=phase,
            crop_size=dataset_opt['crop_size'],
            min_max=(-1, 1),
            structure_path=structure_path,
            function_path=function_path,
            is_crop=dataset_opt['is_crop'],
        )
    elif fusion_mode == 'MEAN':
        from data.Fusion_dataset import FusionDataset_Digtal
        dataset = FusionDataset_Digtal(
            split=phase,
            crop_size=dataset_opt['crop_size'],
            min_max=(-1, 1),
            structure_path=structure_path,
            function_path=function_path,
            is_crop=dataset_opt['is_crop'],
        )
    else:
        raise ValueError(f"Unsupported fusion mode for {fusion_task}: {fusion_mode}")

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
