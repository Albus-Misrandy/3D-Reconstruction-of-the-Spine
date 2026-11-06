"""
Standalone Tool: Clinical Cohort Few-Shot Fine-Tuning CLI
Fine-tunes the CTSpine1K pre-trained nnU-Net on small clinical patient datasets.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from spine_segmentation import finetune_on_clinical_cohort


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CTSpine1K pre-trained nnU-Net on small clinical datasets"
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        required=True,
        help="临床数据集根目录 (需包含 imagesTr/ 和 labelsTr/)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="finetuned_models",
        help="微调权重保存目录 (默认: finetuned_models)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=25,
        help="微调训练轮数 (默认: 25)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="微调学习率 (默认: 1e-4)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=2,
        help="批次大小 (默认: 2)"
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="是否冻结编码器权重，仅微调解码器头"
    )

    args = parser.parse_args()

    finetune_on_clinical_cohort(
        clinical_data_dir=args.data_dir,
        output_checkpoint_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder
    )


if __name__ == "__main__":
    main()
