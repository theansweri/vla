# Copyright (c) Opendatalab. All rights reserved.
import copy
import warnings
import numpy as np
import torch
from loguru import logger
from pathlib import Path

from utils.ocr_utils import (
    check_img,
    preprocess_image,
    sorted_boxes,
    merge_det_boxes,
    update_det_boxes,
    get_rotate_crop_image,
)
from infer.predict_system import TextSystem
from infer import pytorchocr_utility as utility
import argparse


latin_lang = [
    "af",
    "az",
    "bs",
    "cs",
    "cy",
    "da",
    "de",
    "es",
    "et",
    "fr",
    "ga",
    "hr",
    "hu",
    "id",
    "is",
    "it",
    "ku",
    "la",
    "lt",
    "lv",
    "mi",
    "ms",
    "mt",
    "nl",
    "no",
    "oc",
    "pi",
    "pl",
    "pt",
    "ro",
    "rs_latin",
    "sk",
    "sl",
    "sq",
    "sv",
    "sw",
    "tl",
    "tr",
    "uz",
    "vi",
    "french",
    "german",
]
arabic_lang = ["ar", "fa", "ug", "ur"]
cyrillic_lang = [
    "ru",
    "rs_cyrillic",
    "be",
    "bg",
    "uk",
    "mn",
    "abq",
    "ady",
    "kbd",
    "ava",  # noqa: E126
    "dar",
    "inh",
    "che",
    "lbe",
    "lez",
    "tab",
]
devanagari_lang = [
    "hi",
    "mr",
    "ne",
    "bh",
    "mai",
    "ang",
    "bho",
    "mah",
    "sck",
    "new",
    "gom",  # noqa: E126
    "sa",
    "bgc",
]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_model_params(lang, config):
    if lang in config["lang"]:
        params = config["lang"][lang]
        det = params.get("det")
        rec = params.get("rec")
        dict_file = params.get("dict")
        return det, rec, dict_file
    else:
        raise Exception(f"Language {lang} not supported")


root_dir = Path(__file__).resolve().parent.parent


class PytorchPaddleOCR(TextSystem):
    def __init__(self, *args, **kwargs):
        parser = utility.init_args()
        args = parser.parse_args(args)

        self.lang = kwargs.get("lang", "ch_lite")

        device = get_device()
        print(f"当前使用设备: {device}")

        models_dir = root_dir / "models"
        kwargs["det_model_path"] = str(models_dir / "ptocr_v5_mobile_det.pth")
        kwargs["rec_model_path"] = str(models_dir / "ptocr_v5_mobile_rec.pth")
        kwargs["rec_char_dict_path"] = str(models_dir / "ppocrv5_dict.txt")
        # kwargs['rec_batch_num'] = 8

        kwargs["device"] = device

        default_args = vars(args)
        default_args.update(kwargs)
        args = argparse.Namespace(**default_args)

        super().__init__(args)

    def ocr(
        self,
        img,
        det=True,
        rec=True,
        mfd_res=None,
        tqdm_enable=False,
    ):
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det is True:
            logger.error("When input a list of images, det must be false")
            exit(0)
        img = check_img(img)
        imgs = [img]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if det and rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, rec_res = self.__call__(img, mfd_res=mfd_res)
                    if not dt_boxes and not rec_res:
                        ocr_res.append(None)
                        continue
                    tmp_res = [
                        [box.tolist(), res] for box, res in zip(dt_boxes, rec_res)
                    ]
                    ocr_res.append(tmp_res)
                return ocr_res
            elif det and not rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, elapse = self.text_detector(img)
                    # logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
                    if dt_boxes is None:
                        ocr_res.append(None)
                        continue
                    dt_boxes = sorted_boxes(dt_boxes)
                    # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
                    dt_boxes = merge_det_boxes(dt_boxes)
                    if mfd_res:
                        dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                    tmp_res = [box.tolist() for box in dt_boxes]
                    ocr_res.append(tmp_res)
                return ocr_res
            elif not det and rec:
                ocr_res = []
                for img in imgs:
                    if not isinstance(img, list):
                        img = preprocess_image(img)
                        img = [img]
                    rec_res, elapse = self.text_recognizer(img, tqdm_enable=tqdm_enable)
                    # logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
                    ocr_res.append(rec_res)
                return ocr_res

    def __call__(self, img, mfd_res=None):
        if img is None:
            logger.debug("no valid image provided")
            return None, None

        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            return None, None
        else:
            pass
            # logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
        dt_boxes = merge_det_boxes(dt_boxes)

        if mfd_res:
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = self.text_recognizer(img_crop_list)
        # logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res
