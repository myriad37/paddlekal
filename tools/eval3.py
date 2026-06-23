from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import paddle
from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
import tools.program as program


def main():
    global_config = config["Global"]
    # build dataloader
    set_signal_handlers()
    valid_dataloader = build_dataloader(config, "Eval", device, logger)

    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                        config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    out_channels_list = {}
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list["CTCLabelDecode"] = char_num
                    out_channels_list["SARLabelDecode"] = char_num + 2
                    out_channels_list["NRTRLabelDecode"] = char_num + 3
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            out_channels_list = {}
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])
    extra_input_models = [
        "SRN",
        "NRTR",
        "SAR",
        "SEED",
        "SVTR",
        "SVTR_LCNet",
        "VisionLAN",
        "RobustScanner",
        "SVTR_HGNet",
    ]
    extra_input = False
    if config["Architecture"]["algorithm"] == "Distillation":
        for key in config["Architecture"]["Models"]:
            extra_input = (
                    extra_input
                    or config["Architecture"]["Models"][key]["algorithm"]
                    in extra_input_models
            )
    else:
        extra_input = config["Architecture"]["algorithm"] in extra_input_models
    if "model_type" in config["Architecture"].keys():
        if config["Architecture"]["algorithm"] == "CAN":
            model_type = "can"
        elif config["Architecture"]["algorithm"] == "LaTeXOCR":
            model_type = "latexocr"
            config["Metric"]["cal_bleu_score"] = True
        elif config["Architecture"]["algorithm"] == "UniMERNet":
            model_type = "unimernet"
            config["Metric"]["cal_bleu_score"] = True
        elif config["Architecture"]["algorithm"] in [
            "PP-FormulaNet-S",
            "PP-FormulaNet-L",
            "PP-FormulaNet_plus-S",
            "PP-FormulaNet_plus-M",
            "PP-FormulaNet_plus-L",
        ]:
            model_type = "pp_formulanet"
            config["Metric"]["cal_bleu_score"] = True
        else:
            model_type = config["Architecture"]["model_type"]
    else:
        model_type = None

    # build metric
    eval_class = build_metric(config["Metric"])
    # amp
    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")
    amp_custom_black_list = config["Global"].get("amp_custom_black_list", [])
    if use_amp:
        AMP_RELATED_FLAGS_SETTING = {
            "FLAGS_cudnn_batchnorm_spatial_persistent": 1,
        }
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get(
            "use_dynamic_loss_scaling", False
        )
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )
        if amp_level == "O2":
            model = paddle.amp.decorate(
                models=model, level=amp_level, master_weight=True
            )
    else:
        scaler = None

    best_model_dict = load_model(
        config, model, model_type=config["Architecture"]["model_type"]
    )
    if len(best_model_dict):
        logger.info("metric in ckpt ***************")
        for k, v in best_model_dict.items():
            logger.info("{}:{}".format(k, v))

    # --- START OF CUSTOM CSV EVALUATION LOOP ---
    logger.info("Starting custom evaluation and saving to CSV...")

    csv_file_path = "evaluation_results.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Batch_Index", "Ground_Truth", "Prediction", "Confidence", "Is_Correct"])

        model.eval()
        with paddle.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_dataloader, desc="Evaluating")):
                images = batch[0]
                labels = batch[1]

                if use_amp:
                    with paddle.amp.auto_cast(
                            custom_black_list=amp_custom_black_list,
                            level=amp_level
                    ):
                        preds = model(images)
                else:
                    preds = model(images)

                post_result = post_process_class(preds, labels)

                eval_class(post_result, batch)

                for idx in range(len(post_result)):
                    pred_info = post_result[idx]

                    if isinstance(pred_info, tuple) or isinstance(pred_info, list):
                        pred_text = pred_info[0]
                        confidence = pred_info[1]
                    elif isinstance(pred_info, dict):
                        pred_text = pred_info.get('text', '')
                        confidence = pred_info.get('score', 0.0)
                    else:
                        pred_text = str(pred_info)
                        confidence = 0.0

                    gt_text = str(labels[idx]) if idx < len(labels) else ""

                    is_correct = "True" if str(pred_text).strip() == gt_text.strip() else "False"

                    writer.writerow([batch_idx, gt_text, pred_text, confidence, is_correct])

    metric = eval_class.get_metric()

    logger.info(f"Per-sample evaluation saved to {csv_file_path}")
    logger.info("metric eval ***************")
    for k, v in metric.items():
        logger.info("{}:{}".format(k, v))


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
