import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Команды для обучения, тестирования и инференса."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Запуск обучения (train.py)")
    train_parser.add_argument(
        "overrides",
        nargs="*",
        help="Переопределение параметров обучения "
        "training.batch_size, training.lr, training.num_workers, training.num_epochs.",
    )

    test_parser = subparsers.add_parser("test", help="Запуск тестирования (test.py)")
    test_parser.add_argument(
        "overrides",
        nargs="*",
        help="Переопределение модели testing.checkpoint_name",
    )

    infer_parser = subparsers.add_parser("infer", help="Запуск инференса (infer.py)")
    infer_parser.add_argument(
        "overrides",
        nargs="*",
        help="Переопределение изображения infer.image_name",
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd = [
            sys.executable,
            "SkinCancerClassification/train.py",
        ] + args.overrides
        subprocess.run(cmd, check=True)

    elif args.command == "test":
        cmd = [
            sys.executable,
            "SkinCancerClassification/test.py",
        ] + args.overrides
        subprocess.run(cmd, check=True)

    elif args.command == "infer":
        cmd = [
            sys.executable,
            "SkinCancerClassification/infer.py",
        ] + args.overrides
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
