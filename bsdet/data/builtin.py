import os
from .meta_voc import register_meta_voc
from .meta_coco import register_meta_coco
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin import register_all_pascal_voc
from .river import register_river


# -------- COCO -------- #
def register_all_coco(root="datasets"):

    METASPLITS = [
        ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root="datasets"):

    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),

        ("voc_2007_trainval_base4", "VOC2007", "trainval", "base4", 4),
        ("voc_2012_trainval_base4", "VOC2012", "trainval", "base4", 4),
        ("voc_2007_test_base4", "VOC2007", "test", "base4", 4),
        ("voc_2007_test_all4", "VOC2007", "test", "base_novel_4", 4),
        ("voc_2007_test_all5", "VOC2007", "test", "base_novel_5", 5),
        ("voc_2007_test_all6", "VOC2007", "test", "base_novel_6", 6),
        ("voc_2007_test_all7", "VOC2007", "test", "base_novel_7", 7),
        ("voc_2007_test_all8", "VOC2007", "test", "base_novel_8", 8),
        ("voc_2007_test_all9", "VOC2007", "test", "base_novel_9", 9),
    ]
    for prefix in ["all", "novel"]:
        for sid in range(1, 10):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        seed = "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


BASE_CLASSES = ('echinus', 'starfish', 'crab', 'coral')
ALL_CLASSES = ('echinus', 'starfish', 'crab', 'coral', 'novel1', 'novel2', 'novel3', )
def register_all_river(root="datasets"):
    BASE_SPLITS = [
        ("river_base_train", "/data/VOCdevkit/online_river", "train"),
    ]
    for name, dirname, split in BASE_SPLITS:
        year = 2007
        register_river(name, os.path.join(root, dirname), split, year, class_names=BASE_CLASSES)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    ALL_SPLITS = [
        ("river_all_train", "/data/VOCdevkit/online_river", "train"),
        ("river_test2", "/home/luyue/Documents/DeFRCN/online_dataset", "test2"),
        ("river_test5", "/home/luyue/Documents/DeFRCN/online_dataset", "test5"),
    ]
    for name, dirname, split in ALL_SPLITS:
        year = 2007
        register_river(name, os.path.join(root, dirname), split, year, class_names=ALL_CLASSES)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

register_all_coco()
register_all_voc()
register_all_river()
# register_all_pascal_voc('datasets')0