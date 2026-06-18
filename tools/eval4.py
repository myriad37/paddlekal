from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile

# Setup directory paths for PaddleOCR execution context
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
    set_signal_handlers()

    # 1. Build Post Process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # 2. Build Model Configuration
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            for key in config["Architecture"]["Models"]:
                if config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead":
                    out_channels_list = {}
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list["CTCLabelDecode"] = char_num
                    out_channels_list["SARLabelDecode"] = char_num + 2
                    out_channels_list["NRTRLabelDecode"] = char_num + 3
                    config["Architecture"]["Models"][key]["Head"]["out_channels_list"] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"]["out_channels"] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":
            out_channels_list = {}
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:
            config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])

    extra_input_models = [
        "SRN", "NRTR", "SAR", "SEED", "SVTR", "SVTR_LCNet",
        "VisionLAN", "RobustScanner", "SVTR_HGNet",
    ]
    extra_input = False
    if config["Architecture"]["algorithm"] == "Distillation":
        for key in config["Architecture"]["Models"]:
            extra_input = extra_input or config["Architecture"]["Models"][key]["algorithm"] in extra_input_models
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
            "PP-FormulaNet-S", "PP-FormulaNet-L", "PP-FormulaNet_plus-S",
            "PP-FormulaNet_plus-M", "PP-FormulaNet_plus-L",
        ]:
            model_type = "pp_formulanet"
            config["Metric"]["cal_bleu_score"] = True
        else:
            model_type = config["Architecture"]["model_type"]
    else:
        model_type = None

    # 3. Build Metric Evaluator
    eval_class = build_metric(config["Metric"])

    # 4. AMP Setup
    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")
    amp_custom_black_list = config["Global"].get("amp_custom_black_list", [])
    if use_amp:
        AMP_RELATED_FLAGS_SETTING = {"FLAGS_cudnn_batchnorm_spatial_persistent": 1}
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get("use_dynamic_loss_scaling", False)
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )
        if amp_level == "O2":
            model = paddle.amp.decorate(models=model, level=amp_level, master_weight=True)
    else:
        scaler = None

    # 5. Load Model Weights Checkpoint
    best_model_dict = load_model(config, model, model_type=config["Architecture"]["model_type"])
    if len(best_model_dict):
        logger.info("metric in ckpt ***************")
        for k, v in best_model_dict.items():
            logger.info("{}:{}".format(k, v))

    # =========================================================================
    # IN-MEMORY DATASET SPLITTING & SEQUENTIAL EVALUATION
    # =========================================================================

    # Read the master label file path from the YAML configuration
    master_label_file = config["Eval"]["dataset"]["label_file_list"][0]

    # Define prefixes for subset classification based on filename strings
    categories = {
        "Synthetic": "synth_",
        "Handwritten (HDD)": "hdd_",
        "Typed": "typed_"
    }

    split_data = {cat: [] for cat in categories}
    split_data["Other / Unmatched"] = []

    logger.info(f"Reading master dataset: {master_label_file}")
    try:
        with open(master_label_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                filename = line.split('\t')[0].lower()

                matched = False
                for cat, prefix in categories.items():
                    if prefix in filename:
                        split_data[cat].append(line)
                        matched = True
                        break
                if not matched:
                    split_data["Other / Unmatched"].append(line)
    except Exception as e:
        logger.error(f"Failed to read master label file: {e}")
        return

    final_results = {}

    # =================================================================
    # Proxy Wrapper to print Prediction, Confidence, and Ground Truth
    # =================================================================
    class EvalLoggerWrapper:
        def __init__(self, base_eval_class, logger):
            self.base_eval_class = base_eval_class
            self.logger = logger

        def __call__(self, preds, batch, **kwargs):
            # 'preds' holds the decoded predictions: [(pred_str, conf), ...]
            # 'batch[1]' contains the actual ground truth labels
            try:
                gt_texts = batch[1]
                for i in range(min(len(preds), len(gt_texts))):
                    # 1. Unpack Prediction and Confidence
                    if isinstance(preds[i], (tuple, list)) and len(preds[i]) >= 2:
                        pred_text, conf = preds[i][0], preds[i][1]
                    else:
                        pred_text, conf = preds[i], 0.0
                        
                    # 2. Unpack Ground Truth
                    gt_text = gt_texts[i]
                    if hasattr(gt_text, 'numpy'):
                        gt_text = gt_text.numpy()
                    
                    # 3. Log the visual comparison to the console
                    self.logger.info(f"[*] Predict: '{pred_text}' | Conf: {conf:.4f} | Correct (GT): '{gt_text}'")
            except Exception:
                pass # Silently continue if a batch structure differs
            
            # 4. Pass the data back to the original metric evaluator
            return self.base_eval_class(preds, batch, **kwargs)

        def get_metric(self):
            return self.base_eval_class.get_metric()

        def reset(self):
            self.base_eval_class.reset()

    # Instantiate the wrapper once
    logging_eval_class = EvalLoggerWrapper(eval_class, logger)

    # Sequentially build evaluation dataloaders for each sub-dataset split
    for cat, lines in split_data.items():
        if not lines:
            continue

        logger.info(f"\n{'=' * 60}\nEvaluating Subset: {cat} ({len(lines)} items)\n{'=' * 60}")

        # Construct a temporary text file to serve as the subset label list
        fd, temp_tsv_path = tempfile.mkstemp(suffix=".tsv", text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as temp_f:
            temp_f.writelines(lines)

        try:
            # Overwrite the active configuration parameter to point to the temporary file
            config["Eval"]["dataset"]["label_file_list"] = [temp_tsv_path]

            # Rebuild the dataloader exclusively targeting the subset items
            subset_dataloader = build_dataloader(config, "Eval", device, logger)

            # Calculate raw metrics using our new logging wrapper
            metric = program.eval(
                model,
                subset_dataloader,
                post_process_class,
                logging_eval_class,
                model_type,
                extra_input,
                scaler,
                amp_level,
                amp_custom_black_list,
            )

            logger.info(f"--- metric eval for {cat} ---")
            for k, v in metric.items():
                logger.info("{}:{}".format(k, v))

            final_results[cat] = metric

        finally:
            # Clean up the dynamic temporary file from disk immediately
            if os.path.exists(temp_tsv_path):
                os.remove(temp_tsv_path)

    # =========================================================================
    # GENERATE AND WRITE PERFORMANCE SUMMARY FILE
    # =========================================================================
    summary_lines = []
    summary_lines.append(f"\n{'=' * 60}\nFINAL SUBSET PERFORMANCE SUMMARY\n{'=' * 60}\n")
    for cat, res in final_results.items():
        acc = res.get('acc', 0.0)
        norm_ed = res.get('norm_edit_dis', 0.0)
        summary_lines.append(f"{cat.ljust(20)} | Accuracy (acc): {acc:.4f} | Char Accuracy (norm_ED): {norm_ed:.4f}\n")
    summary_lines.append("=" * 60 + "\n")

    # Output matrix verification to logs
    for line in summary_lines:
        logger.info(line.strip())

    # Build destination path inside the designated model tracking directory
    output_summary_path = os.path.join(global_config["save_model_dir"], "subset_evaluation_summary.txt")
    try:
        os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
        with open(output_summary_path, "w", encoding="utf-8") as sf:
            sf.writelines(summary_lines)
        logger.info(f"Successfully saved permanent evaluation file to: {output_summary_path}")
    except Exception as e:
        logger.error(f"Failed to generate output metrics file: {e}")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
