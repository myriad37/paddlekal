from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import csv

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


class PredExtractorMetric:
    """
    A transparent wrapper around PaddleOCR's metric class to intercept
    and store per-image predictions for CSV generation. Handles scalar
    and sequence/tuple confidence scores safely.
    """

    def __init__(self, base_metric):
        self.base_metric = base_metric
        self.csv_rows = []

    def __call__(self, preds, batch, **kwargs):
        # Extract batch predictions depending on architecture (e.g., Distillation vs Standard)
        if isinstance(preds, dict):
            eval_key = next((k for k in preds.keys() if 'Student' in k), list(preds.keys())[0])
            batch_preds = preds[eval_key]
        else:
            batch_preds = preds

        # Normalize prediction formats to (text, confidence) safely
        for i, pred_item in enumerate(batch_preds):
            text = ""
            conf = 0.0

            if isinstance(pred_item, (tuple, list)) and len(pred_item) >= 2:
                text = pred_item[0]
                raw_conf = pred_item[1]

                # Handle cases where confidence is a tuple/list (e.g., token-level scores)
                if isinstance(raw_conf, (tuple, list)):
                    if len(raw_conf) > 0:
                        try:
                            # Calculate average token confidence across the sequence
                            conf = sum(float(x) for x in raw_conf) / len(raw_conf)
                        except (ValueError, TypeError):
                            try:
                                # Fallback to the first element if it's a nested structure
                                conf = float(raw_conf[0])
                            except (ValueError, TypeError, IndexError):
                                conf = 0.0
                    else:
                        conf = 0.0
                else:
                    try:
                        conf = float(raw_conf)
                    except (ValueError, TypeError):
                        conf = 0.0

            elif isinstance(pred_item, dict) and 'text' in pred_item:
                text = pred_item['text']
                raw_conf = pred_item.get('score', 0.0)
                if isinstance(raw_conf, (tuple, list)) and len(raw_conf) > 0:
                    try:
                        conf = sum(float(x) for x in raw_conf) / len(raw_conf)
                    except:
                        conf = 0.0
                else:
                    try:
                        conf = float(raw_conf)
                    except:
                        conf = 0.0
            else:
                text = str(pred_item)
                conf = 0.0

            self.csv_rows.append((str(text), float(conf)))

        # Forward the data to the original metric calculator
        self.base_metric(preds, batch, **kwargs)

    def reset(self):
        self.csv_rows = []
        if hasattr(self.base_metric, 'reset'):
            return self.base_metric.reset()

    def get_metric(self):
        if hasattr(self.base_metric, 'get_metric'):
            return self.base_metric.get_metric()
        return {}

    def __getattr__(self, name):
        return getattr(self.base_metric, name)


def main():
    global_config = config["Global"]
    set_signal_handlers()

    # 1. Build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # 2. Build model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in ["Distillation"]:  # distillation model
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
        "SRN", "NRTR", "SAR", "SEED", "SVTR", "SVTR_LCNet",
        "VisionLAN", "RobustScanner", "SVTR_HGNet",
    ]
    extra_input = False
    if config["Architecture"]["algorithm"] == "Distillation":
        for key in config["Architecture"]["Models"]:
            extra_input = (extra_input or config["Architecture"]["Models"][key]["algorithm"] in extra_input_models)
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

    # 3. Build metric & Wrap it for Prediction Extraction
    base_eval_class = build_metric(config["Metric"])
    eval_class = PredExtractorMetric(base_eval_class)

    # 4. AMP
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

    best_model_dict = load_model(config, model, model_type=config["Architecture"]["model_type"])
    if len(best_model_dict):
        logger.info("metric in ckpt ***************")
        for k, v in best_model_dict.items():
            logger.info("{}:{}".format(k, v))

    # =========================================================================
    # IN-MEMORY DATASET SPLITTING & SEQUENTIAL EVALUATION
    # =========================================================================
    master_label_file = config["Eval"]["dataset"]["label_file_list"][0]

    categories = {
        "Synthetic": "synth",
        "Handwritten (HDD)": "hdd_",
        "Handwritten": "handwritten_",
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

    # Setup CSV Writer
    csv_file_path = os.path.join(global_config.get("save_model_dir", "./"), "test_predictions.csv")
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    final_results = {}

    with open(csv_file_path, "w", encoding="utf-8", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["image_name", "ground_truth", "prediction", "confidence", "correct", "subset"])

        for cat, lines in split_data.items():
            if not lines:
                continue

            logger.info(f"\n{'=' * 60}\nEvaluating Subset: {cat} ({len(lines)} items)\n{'=' * 60}")

            fd, temp_tsv_path = tempfile.mkstemp(suffix=".tsv", text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as temp_f:
                temp_f.writelines(lines)

            try:
                config["Eval"]["dataset"]["label_file_list"] = [temp_tsv_path]
                config["Eval"]["loader"]["shuffle"] = False
                config["Eval"]["loader"]["drop_last"] = False
                subset_dataloader = build_dataloader(config, "Eval", device, logger)

                # Reset the wrapper row container before each split runs
                eval_class.reset()

                # Execute evaluation
                metric = program.eval(
                    model, subset_dataloader, post_process_class, eval_class,
                    model_type, extra_input, scaler, amp_level, amp_custom_black_list,
                )

                logger.info(f"--- metric eval for {cat} ---")
                for k, v in metric.items():
                    logger.info("{}:{}".format(k, v))

                final_results[cat] = metric

                # Write Predictions to CSV
                for line, (pred_text, conf) in zip(lines, eval_class.csv_rows):
                    parts = line.strip().split('\t')
                    image_name = parts[0]
                    gt_text = parts[1] if len(parts) > 1 else ""

                    # Compute correctness (Exact match, space-stripped)
                    correct = (str(gt_text).strip() == str(pred_text).strip())

                    csv_writer.writerow([image_name, gt_text, pred_text, f"{conf:.4f}", correct, cat])

            finally:
                if os.path.exists(temp_tsv_path):
                    os.remove(temp_tsv_path)

    logger.info(f"Prediction CSV successfully saved to: {csv_file_path}")

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

    for line in summary_lines:
        logger.info(line.strip())

    output_summary_path = os.path.join(global_config.get("save_model_dir", "./"), "subset_evaluation_summary.txt")
    try:
        with open(output_summary_path, "w", encoding="utf-8") as sf:
            sf.writelines(summary_lines)
        logger.info(f"Summary file saved to: {output_summary_path}")
    except Exception as e:
        logger.error(f"Failed to generate output metrics file: {e}")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
