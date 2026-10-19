"""Generate datasets from the supplied configurations; no training."""
from pathlib import Path
import argparse
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def generate(output, small=None):
    from fmcw_dataset.config import validate
    from fmcw_dataset.storage import write_dataset
    output = Path(output)
    for source_group, target_group in [('principal', 'principal'), ('external', 'externe')]:
        target = output / target_group
        target.mkdir(parents=True, exist_ok=True)
        files = []
        for path in sorted((ROOT/'fmcw_dataset/configs'/source_group).glob('*.json')):
            config = json.loads(path.read_text())
            if small is not None:
                config['count'] = small
            config = validate(config)
            destination = target / (config['name'] + '.h5')
            if destination.exists():
                raise FileExistsError(f'Choose another output directory: {destination}')
            print(f"Generating {target_group}/{destination.name}: {config['count']} scenes", flush=True)
            write_dataset(config, target)
            files.append(destination)
        archive = output / (target_group + '.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as stream:
            for path in files:
                stream.write(path, path.name)
        print(f'Created {archive}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'generated')
    parser.add_argument('--small', type=int, help='Override scene count per configuration.')
    args = parser.parse_args()
    if args.small is not None and args.small < 1:
        parser.error('--small must be positive')
    generate(args.output, args.small)


if __name__ == '__main__':
    main()
