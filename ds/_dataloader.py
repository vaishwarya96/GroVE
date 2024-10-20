"""libaray for multi-modal dataset loaders.

Acknowledgements:
`image_to_caption_collate_fn` is based on
https://github.com/yalesong/pvse/blob/master/data.py
"""
import os
from os.path import join as ospj

import numpy as np
from PIL import Image
from typing import Union, List

import torch
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import transforms
import random

from ds.flickr import FlickrCap, FlickrBboxes
from ds.coco import CocoCaptionsCap, CocoBboxes
from ds.cub import CUBCaption, CUBSampler, CUBData
from ds.fashion200k import Fashion200k,BaseDataset
from ds.flo import FLOCaption, FLOSampler
from ds.vocab import Vocabulary
#from datasets._transforms import imagenet_transform
from ds._transforms import  caption_transform
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from config import get_cfg_defaults 

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

cfg = get_cfg_defaults()

def imagenet_normalize():
    """Standard ImageNet normalize transform
    """
#     return transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
    return transforms.Normalize(
        mean=cfg.DATASET.IMG_MEAN,
        std=cfg.DATASET.IMG_STD)
def imagenet_transform_fn(resize_size=cfg.DATASET.IMG_SIZE,
                       crop_size=cfg.DATASET.IMG_SIZE,
                       random_resize_crop=False,
                       random_erasing_prob=cfg.DATASET.IMG_MASKING_PROB,
                       custom_transforms=None):
    """Standard ImageNet transform with resize/crop/normalize.

    Args:
        resize_size (int, Default: 256): resize for validation
            (only used when random_resize_crop is False).
        crop_size (int, Default: 224): final crop size.
        random_resize_crop (bool, Default: False): if True, use random transform (for training),
            if False, use center crop (for validation).
        custom_transforms (list of transform, Default: None): additional transforms.
    """
    if custom_transforms is not None:
        if not isinstance(custom_transforms, list):
            raise TypeError(f'custom_transforms should be list, not {type(custom_transforms)}')
    transform = []
    if random_resize_crop:
        transform.append(transforms.RandomResizedCrop(crop_size))
        transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.Resize(resize_size))
        transform.append(transforms.CenterCrop(crop_size))
    transform.append(transforms.ToTensor())
    transform.append(imagenet_normalize())

    if custom_transforms:
        transform.extend(custom_transforms)

    if random_erasing_prob > 0:
        print(f'adding cutout {random_erasing_prob}')
        transform.append(transforms.RandomErasing(random_erasing_prob))
                                        #mode='const',
                                        #max_count=1, num_splits=0, device='cpu'))
    transform.append(transforms.RandomErasing(random_erasing_prob))

    transform = transforms.Compose(transform)
    #print("Transform Called")
    return transform



imagenet_transform = imagenet_transform_fn()
_tokenizer = _Tokenizer()


def mask_tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True, mask_prob: float=0.3) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if mask_prob > 0:
        mask_token = "[MASK]"
        words = texts.split()
        masked_words = [(mask_token if random.random() < mask_prob else word) for word in words]
        texts = " ".join(masked_words)



    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True, mask_prob: float=0.0) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def image_to_caption_collate_fn(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
      data: list of (image, sentence) tuple.
        - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
        - sentence: torch tensor of shape (?); variable length.

    Returns:
      images: torch tensor of shape (batch_size, 3, 256, 256) or
              (batch_size, padded_length, 3, 256, 256).
      targets: torch tensor of shape (batch_size, padded_length).
      lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, ann_ids, image_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [len(cap) for cap in sentences]
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    cap_lengths = torch.Tensor(cap_lengths).long()
    return images, targets, cap_lengths, ann_ids, image_ids


def load_vocab(vocab_path):
    if isinstance(vocab_path, str):
        vocab = Vocabulary()
        vocab.load_from_pickle(vocab_path)
    else:
        vocab = vocab_path
    return vocab


def _get_cub_file_paths(dataset_name, dataset_root, caption_root):
    """Select proper train / val classes and omit id files.
    The split is based on CVPR'17 Zero-Shot Learning -- The Good, the Bad and the Ugly
    See more details in
    https://arxiv.org/abs/1703.04394
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly
    Args:
      dataset_name: name of dataset
        - cub_trainval{idx} (idx in [1, 2, 3]):
            3-fold validation splits to search hyperparameters.
            Each split contains 100 train classes / 50 validation classes.
        - cub:
            The final split used for the final benchmark.
            This split conntains 150 train classes / 50 unseen test classes (not in trainval)
    """
    if dataset_name == 'cub_trainval1':
        train_classes = './ds/annotations/cub/trainclasses1.txt'
        val_classes = './ds/annotations/cub/valclasses1.txt'
        omit_ids = './ds/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub_trainval2':
        train_classes = './ds/annotations/cub/trainclasses2.txt'
        val_classes = './ds/annotations/cub/valclasses2.txt'
        omit_ids = './ds/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub_trainval3':
        train_classes = './ds/annotations/cub/trainclasses3.txt'
        val_classes = './ds/annotations/cub/valclasses3.txt'
        omit_ids = './ds/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub':
        train_classes = './ds/annotations/cub/trainvalclasses.txt'
        val_classes = './ds/annotations/cub/testclasses.txt'
        omit_ids = './ds/annotations/cub/seen_test_images.txt'
    else:
        raise ValueError(f'Invalide dataset_name: {dataset_name}')

    image_root = os.path.join(dataset_root, 'images/')

    return train_classes, val_classes, omit_ids, image_root, caption_root


def _get_cub_loader(image_root, caption_root,
                    data_classes, 
                    train=False,
                    omit_ids=None,
                    ids=None,
                    ):


    if cfg.DATASET.TEXT_MASKING_PROB > 0.0:
        _caption_transform = mask_tokenize #tokenize #mask_tokenize
    else:
        _caption_transform = tokenize
    
    _image_transform = imagenet_transform_fn(
        random_resize_crop=train,
        random_erasing_prob=cfg.DATASET.IMG_MASKING_PROB,
        )

    cub_dataset = CUBCaption(image_root, caption_root,  
                             data_classes,
                             #transform=_image_transform,
                             _image_transform,
                             #caption_transform(vocab, caption_drop_prob),
                             omit_ids=omit_ids,
                             target_transform=_caption_transform,
                             ids=ids)
    if train:

        #sampler = CUBSampler(cub_dataset, len(cub_dataset.target_classes))
        dataloader = DataLoader(cub_dataset,
                                batch_size=cfg.DATASET.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS,
#                                collate_fn=image_to_caption_collate_fn,
                                pin_memory=True)
        #dataloader = DataLoader(cub_dataset, batch_sampler=sampler,
        #                        num_workers=num_workers,
#       #                         collate_fn=image_to_caption_collate_fn,
        #                        pin_memory=True)
    else:
        dataloader = DataLoader(cub_dataset,
                                batch_size=cfg.DATASET.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS,
#                                collate_fn=image_to_caption_collate_fn,
                                pin_memory=True)
    print(f'Loading CUB Caption: n_images {cub_dataset.n_images} n_captions {len(cub_dataset.targets)}...')
    return cub_dataset, dataloader


def prepare_cub_dataloaders(dataset_root,
                            caption_root,
                            dataset_name='cub',
                            ):
    """Prepare CUB Caption train / val / test dataloaders
    CUB Caption loader has a fixed batch size
    - train loader: # classes (trainval = 100, full = 150)
    - test loader: 64 (hard coded at L#203)
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_name (str): name of dataset
            - cub_trainval{idx} (idx in [1, 2, 3]):
                3-fold validation splits to search hyperparameters.
                Each split contains 100 train classes / 50 validation classes.
            - cub:
                The final split used for the final benchmark.
                This split conntains 150 train classes / 50 unseen test classes (not in trainval)
        dataset_root (str): root of your CUB images (see README.md for detailed dataset hierarchy)
        caption_root (str): root of your CUB captions (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/cub_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "val_in"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """


    train_classes, val_classes, omit_ids, image_root, caption_root = _get_cub_file_paths(
        dataset_name, dataset_root, caption_root)

    dataloaders = {}
    data = {}
    data['train'], dataloaders['train'] = _get_cub_loader(
        image_root, caption_root,
        train_classes,
        train=True,
        omit_ids=omit_ids,
    )

    data['test'], dataloaders['test'] = _get_cub_loader(
        image_root, caption_root,
        val_classes,
        train=False,
    )

    data['val'], dataloaders['val'] = _get_cub_loader(
        image_root, caption_root,
        train_classes,
        train=False,
        ids=omit_ids,
    )

    return data, dataloaders



def _get_coco_loader(image_root,
                     annotation_path,
                     ids, 
                     train=False,
                     extra_ids=None,
                     extra_annotation_path=None,
                    ):
    

    if cfg.DATASET.TEXT_MASKING_PROB > 0.0:
        _caption_transform = mask_tokenize #tokenize #mask_tokenize
    else:
        _caption_transform = tokenize
    
    _image_transform = imagenet_transform_fn(
        random_resize_crop=train,
        random_erasing_prob=cfg.DATASET.IMG_MASKING_PROB,
        )
    

    coco_dataset = CocoCaptionsCap(image_root, annotation_path,
                                   extra_annFile=extra_annotation_path,
                                   ids=ids,
                                   extra_ids=extra_ids,
                                   transform=_image_transform,
                                   target_transform=_caption_transform)

    dataloader = DataLoader(coco_dataset,
                            batch_size=cfg.DATASET.BATCH_SIZE,
                            shuffle=train,
                            num_workers=cfg.SYSTEM.NUM_WORKERS,
                            pin_memory=True)

    print(f'Loading COCO Caption: n_images {coco_dataset.n_images} n_captions {len(coco_dataset)}...')
    return coco_dataset, dataloader


def _get_coco_file_paths(dataset_root):
    """Select proper train / val classes and omit id files.
    """
    train_ids = np.load('./ds/annotations/coco/data/coco/annotations/coco_train_ids.npy')
    train_extra_ids = np.load('./ds/annotations/coco/data/coco/annotations/coco_restval_ids.npy')
    val_ids = np.load('./ds/annotations/coco/data/coco/annotations/coco_dev_ids.npy')[:5000]
    te_ids = np.load('./ds/annotations/coco/data/coco/annotations/coco_test_ids.npy')

#     image_root = os.path.join(dataset_root, 'images/trainval35k')
    image_root = os.path.join(dataset_root, 'images/tmp') # train + valid
    train_ann = os.path.join(dataset_root, 'captions_train2014.json')
    val_ann = os.path.join(dataset_root, 'captions_val2014.json')

    return train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann

def prepare_coco_dataloaders(
                             dataset_root,
                             ):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """


    train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann = _get_coco_file_paths(dataset_root)

    dataloaders = {}
    data = {}

    data['train'], dataloaders['train'] = _get_coco_loader(
        image_root, train_ann, train_ids,
        train=True,
        extra_annotation_path=val_ann,
        extra_ids=train_extra_ids,
    )

    data['val'], dataloaders['val'] = _get_coco_loader(
        image_root, val_ann, val_ids,
        train=False,
    )

    data['test'], dataloaders['test'] = _get_coco_loader(
        image_root, val_ann, te_ids, 
        train=False,
    )

    return data, dataloaders

def _get_flickr_file_paths(dataset_root):
    image_root = ospj(dataset_root, 'flickr30k_images')
    train_ids_path = './ds/annotations/flickr/train.txt'
    valid_ids_path = './ds/annotations/flickr/val.txt'
    test_ids_path = './ds/annotations/flickr/test.txt'
    return image_root, train_ids_path, valid_ids_path, test_ids_path

def _get_flickr_loader(image_root,
                     image_ids_path,
                     train=False,
                     ):
    
    
    if cfg.DATASET.TEXT_MASKING_PROB > 0.0:
        _caption_transform = mask_tokenize #tokenize #mask_tokenize
    else:
        _caption_transform = tokenize
    
    _image_transform = imagenet_transform_fn(
        random_resize_crop=train,
        random_erasing_prob=cfg.DATASET.IMG_MASKING_PROB,
        )
        

    flickr_dataset = FlickrCap(image_root, image_ids_path,
                             transform=_image_transform,
                             target_transform=_caption_transform)

    dataloader = DataLoader(flickr_dataset,
                            batch_size=cfg.DATASET.BATCH_SIZE,
                            shuffle=train,
                            num_workers=cfg.SYSTEM.NUM_WORKERS,
                            pin_memory=True)
    print(f'Loading Flickr Caption: n_captions {len(flickr_dataset)}...')
    return flickr_dataset, dataloader

def prepare_flickr_dataloaders(
                               dataset_root,
                               ):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    
    image_root, train_ids_path, valid_ids_path, test_ids_path = _get_flickr_file_paths(dataset_root)

    dataloaders = {}
    data = {}

    data['train'], dataloaders['train'] = _get_flickr_loader(
        image_root,
        image_ids_path=train_ids_path,
        train=True,
    )

    data['val'], dataloaders['val'] = _get_flickr_loader(
        image_root,
        image_ids_path=valid_ids_path,
        train=False,
    )

    data['test'], dataloaders['test'] = _get_flickr_loader(
        image_root,
        image_ids_path=test_ids_path,
        train=False,
    )

    return data, dataloaders


def _get_flo_file_paths(dataset_name, dataset_root, caption_root):
    """Select proper train / val classes and omit id files.
    The split is based on CVPR'17 Zero-Shot Learning -- The Good, the Bad and the Ugly
    See more details in
    https://arxiv.org/abs/1703.04394
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly
    Args:
      dataset_name: name of dataset
        - cub_trainval{idx} (idx in [1, 2, 3]):
            3-fold validation splits to search hyperparameters.
            Each split contains 100 train classes / 50 validation classes.
        - flo:
            The final split used for the final benchmark.
            This split conntains 82 train classes / 20 unseen test classes (not in trainval)
    """

    if dataset_name == 'flo':
        train_classes = './ds/annotations/flo/trainvalclasses.txt'
        val_classes = './ds/annotations/flo/testclasses.txt'

    else:
        raise ValueError(f'Invalide dataset_name: {dataset_name}')

    image_root = dataset_root

    return train_classes, val_classes, image_root, caption_root


def _get_flo_loader(image_root, caption_root,
                    data_classes,
                    train=False):



    if cfg.DATASET.TEXT_MASKING_PROB > 0.0:
        _caption_transform = mask_tokenize #tokenize #mask_tokenize
    else:
        _caption_transform = tokenize
    
    _image_transform = imagenet_transform_fn(
        random_resize_crop=train,
        random_erasing_prob=cfg.DATASET.IMG_MASKING_PROB,
        )

    flo_dataset = FLOCaption(image_root, caption_root,
                             data_classes,
                             imagenet_transform,
                             target_transform=_caption_transform,)
    if train:
        sampler = FLOSampler(flo_dataset, len(flo_dataset.target_classes))
        dataloader = DataLoader(flo_dataset, batch_sampler=sampler,
                                pin_memory=True)
    else:
        dataloader = DataLoader(flo_dataset,
                                shuffle=False,
                                pin_memory=True)
    print(f'Loading FLO Caption: n_images {flo_dataset.n_images} n_captions {len(flo_dataset.targets)}...')
    return dataloader, flo_dataset


def prepare_flo_dataloaders(
                            dataset_root,
                            caption_root,
                            dataset_name='flo',
                            ):
    """Prepare FLO Caption train / val / test dataloaders
    FLO Caption loader has a fixed batch size
    - train loader: # classes (trainval = 100, full = 150)
    - test loader: 64 (hard coded at L#203)
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_name (str): name of dataset
            - flo_trainval{idx} (idx in [1, 2, 3]):
                3-fold validation splits to search hyperparameters.
                Each split contains 100 train classes / 50 validation classes.
            - cub:
                The final split used for the final benchmark.
                This split conntains 150 train classes / 50 unseen test classes (not in trainval)
        dataset_root (str): root of your CUB images (see README.md for detailed dataset hierarchy)
        caption_root (str): root of your CUB captions (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/cub_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "val_in"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    train_classes, val_classes, image_root, caption_root = _get_flo_file_paths(
        dataset_name, dataset_root, caption_root)


    dataloaders = {}
    dataset = {}
    dataloaders['train'], dataset['train'] = _get_flo_loader(
        image_root, caption_root,
        train_classes,
        train=True,
    )

    dataloaders['test'], dataset['test'] = _get_flo_loader(
        image_root, caption_root,
        val_classes,
        train=False,
    )

    dataloaders['val'], dataset['val'] = _get_flo_loader(
        image_root, caption_root,
        train_classes,
        train=False,
    )

    return dataset, dataloaders
