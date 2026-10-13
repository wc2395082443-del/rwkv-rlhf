import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


RELEASE_BASE = 'https://github.com/wc2395082443-del/rwkv-rlhf/releases/download/g1k-step1000-backup-20261013-v1/'
BLOCK = 8 * 1024 ** 2


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(BLOCK):
            result.update(chunk)
    return result.hexdigest()


def checked_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe archive path: ' + name)
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes destination: ' + name)
    return target


def download(directory, name):
    target = checked_path(directory, name)
    if target.exists():
        return target
    temporary = target.with_name(target.name + '.partial')
    request = urllib.request.Request(RELEASE_BASE + urllib.parse.quote(name), headers={'User-Agent': 'RWKV-backup-restore'})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output, BLOCK)
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description='Verify and restore the isolated G1K GitHub backup. Full extraction requires Linux.')
    parser.add_argument('--assets-dir', type=Path, default=Path('backup-assets'))
    parser.add_argument('--output-root', type=Path, default=Path('restored'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--packages', nargs='*', help='Optional logical package names from backup_manifest.json')
    args = parser.parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(args.assets_dir, 'backup_manifest.json')
    manifest = json.loads((args.assets_dir / 'backup_manifest.json').read_text(encoding='utf-8'))
    packages = [package for package in manifest['packages'] if not args.packages or package['name'] in args.packages]
    if args.packages and len(packages) != len(set(args.packages)):
        raise ValueError('Unknown package requested')
    if not args.verify_only:
        if os.name != 'posix':
            raise RuntimeError('Use Linux for extraction to preserve original POSIX filenames. --verify-only works on Windows.')
        args.output_root.mkdir(parents=True, exist_ok=True)
    for package in packages:
        for part in package['parts']:
            path = download(args.assets_dir, part['name']) if args.download else args.assets_dir / part['name']
            if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                raise ValueError('Asset checksum mismatch: ' + part['name'])
        combined = args.assets_dir / package['name']
        if len(package['parts']) != 1 or package['parts'][0]['name'] != combined.name:
            temporary = combined.with_name(combined.name + '.assembling')
            with temporary.open('wb') as output:
                for part in package['parts']:
                    with (args.assets_dir / part['name']).open('rb') as stream:
                        shutil.copyfileobj(stream, output, BLOCK)
            temporary.replace(combined)
        if combined.stat().st_size != package['bytes'] or digest(combined) != package['sha256']:
            raise ValueError('Reassembled package checksum mismatch: ' + package['name'])
        if not args.verify_only:
            if package['kind'] == 'checkpoint':
                target = checked_path(args.output_root, package['source_path'].lstrip('/'))
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if digest(target) != package['sha256']:
                        raise FileExistsError('Refusing to overwrite different checkpoint: ' + str(target))
                else:
                    shutil.copyfile(combined, target)
            else:
                with tarfile.open(combined, 'r:gz') as archive:
                    for member in archive:
                        target = checked_path(args.output_root, member.name)
                        if not member.isfile():
                            raise ValueError('Unexpected non-file archive member: ' + member.name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source:
                            if target.exists():
                                expected = hashlib.sha256()
                                while chunk := source.read(BLOCK):
                                    expected.update(chunk)
                                if digest(target) != expected.hexdigest():
                                    raise FileExistsError('Refusing to overwrite different existing file: ' + str(target))
                            else:
                                with target.open('wb') as output:
                                    shutil.copyfileobj(source, output, BLOCK)
                                target.chmod(member.mode & 0o777)
        print('VERIFIED', package['name'], package['sha256'])


if __name__ == '__main__':
    main()
