def load_data(data, data_root, data_imsize, is_eval=False):
    # Data loading
    
    collate_fn = None
    
    if data == "clevrtex_full" or data == "clevrtex_camo" or data == "clevrtex_outd":
        from source.data.datasets.objs.clevr_tex import get_clevrtex_pair, get_clevrtex, collate_fn
        if data == "clevrtex_camo":
            assert is_eval, "Camo dataset is only for evaluation"
            data_type = "camo"
            default_path = "./data/clevr_tex/clevrtex_camo"
        elif data == "clevrtex_outd":
            assert is_eval, "OOD dataset is only for evaluation"
            data_type = "outd"
            default_path = "./data/clevr_tex/clevrtex_outd"
        else:
            data_type = "full"
            default_path = "./data/clevr_tex/clevrtex_full"
            
        data_root = (
            default_path
            if data_root is None
            else data_root
        )
        imsize = 128 if data_imsize is None else data_imsize
        if is_eval:
            dataset = get_clevrtex(data_root, split="test", data_type=data_type, imsize=imsize, return_meta_data=True)
        else:
            dataset = get_clevrtex_pair(
                root=data_root,
                split="train",
            )
    elif data == "clevr":
        from source.data.datasets.objs.clevr import get_clevr_pair, get_clevr
        imsize = 128 if data_imsize is None else data_imsize
        data_root = "./data/clevr_with_masks/"
        sstrainset = get_clevr_pair(
            root="./data/clevr_with_masks/",
            split="train",
        )

   
    elif data == "imagenet":
        from source.data.datasets.objs.imagenet import get_imagenet_pair
        data_root = (
            "./data/ImageNet2012/"
            if data_root is None
            else data_root
        )
        sstrainset = get_imagenet_pair(
            root=data_root,
            split="train",
            hflip=True,
            imsize=256 if data_imsize is None else data_imsize,
        )
    elif data == "coco":
        imsize = 256 if data_imsize is None else data_imsize
        from source.data.datasets.objs.coco import get_coco_pair
        data_root = "./data/COCO"
        sstrainset = get_coco_pair(root=data_root)
        
    elif data == "dsprites":
        imsize = 64 if data_imsize is None else data_imsize
        from source.data.datasets.objs.dsprites import get_dsprites_pair
        sstrainset = get_dsprites_pair(
            root="./data/multi_dsprites/",
            split="train",
            imsize=imsize,
        )
        
    elif data == "tetrominoes":
        from source.data.datasets.objs.tetrominoes import get_tetrominoes_pair

        imsize = 32 if data_imsize is None else data_imsize
        sstrainset = get_tetrominoes_pair(
            root="./data/tetrominoes/",
            split="train",
            imsize=imsize,
        )
        
    elif data == "Shapes":
        imsize = 40 if data_imsize is None else data_imsize
        from source.data.datasets.objs.shapes import get_shapes_pair

        dataset = get_shapes_pair(
            root="./data/Shapes/",
            split="train",
            imsize=imsize,
        )
    return dataset, imsize, collate_fn

