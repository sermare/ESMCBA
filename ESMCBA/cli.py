# ESMCBA/cli.py

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from .registry import get_model_filename_for_hla
from .embeddings_generation_main import run_embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ESMCBA embeddings for a set of peptides."
    )
    parser.add_argument(
        "--hla",
        required=True,
        help="HLA allele, e.g. B1402 or HLA-B*14:02",
    )
    parser.add_argument(
        "--encoding",
        default="epitope",
        help="Encoding type (default: epitope). Use 'hla' to prepend HLA sequence.",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Directory to save outputs (default: ./outputs)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Base name for output files. Default is <HLA>-ESMCBA",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for inference (default: 10)",
    )
    parser.add_argument(
        "--umap_dims",
        type=int,
        default=2,
        choices=[2, 3],
        help="UMAP dimensionality for visualization (default: 2)",
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--peptides",
        nargs="+",
        required=True,
        help="Peptides to evaluate, separated by spaces.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Map HLA to checkpoint filename
    model_filename = get_model_filename_for_hla(args.hla)

    # 2. Download checkpoint from Hugging Face Hub (into local cache)
    model_path = hf_hub_download(
        repo_id="smares/ESMCBA",
        filename=model_filename,
        repo_type="model",
    )

    # 3. Choose output name
    name = args.name or f"{args.hla}-ESMCBA"

    # 4. Make sure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 5. Run embeddings pipeline
    run_embeddings(
        model_path=model_path,
        name=name,
        hla=args.hla,
        encoding=args.encoding,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        peptides=args.peptides,
        umap_dims=args.umap_dims,
        umap_neighbors=args.umap_neighbors,
    )


if __name__ == "__main__":
    main()

