##

import argparse
from deployment.generator import generate_terraform


def main():
    parser = argparse.ArgumentParser(
        description="AutoMLOps Suite Terraform Generator"
    )

    parser.add_argument(
        "--provider",
        choices=["aws", "gcp"],
        required=True,
        help="Cloud provider"
    )

    parser.add_argument(
        "--description",
        required=True,
        help="Infrastructure description"
    )

    parser.add_argument(
        "--output",
        default="generated",
        help="Output directory"
    )

    args = parser.parse_args()

    try:
        file_path = generate_terraform(
            provider=args.provider,
            description=args.description,
            output_dir=args.output
        )
    except Exception as error:
        parser.exit(1, f"Deployment generation failed: {error}\n")

    print(f"Terraform configuration generated: {file_path}")
    print("Review and validate the Terraform before deploying.")


if __name__ == "__main__":
    main()
