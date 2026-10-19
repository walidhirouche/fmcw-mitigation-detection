"""Launch the FMCW dataset generator."""
import sys


def main():
    try:
        from fmcw_dataset.server import main as run
    except ModuleNotFoundError as exc:
        sys.exit(f"Missing module: {exc.name}. Run: python -m pip install -r requirements.txt")
    run()


if __name__ == '__main__':
    main()
