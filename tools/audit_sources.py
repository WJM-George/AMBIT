"""Static source audit; does not import models or execute project launchers."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache'}


def main():
    rows = []
    errors = []
    absolute_paths = []
    checked = {'python': 0, 'shell': 0}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(p in SKIP for p in path.relative_to(ROOT).parts):
            continue
        if path.suffix not in {'.py', '.sh'}:
            continue
        relative = str(path.relative_to(ROOT))
        if path.is_symlink():
            errors.append({'path': relative, 'error': 'source is a symlink'})
            continue
        content = path.read_bytes()
        text = content.decode('utf-8')
        rows.append({'path': relative, 'bytes': len(content),
                     'lines': len(text.splitlines()), 'sha256': hashlib.sha256(content).hexdigest()})
        if '/mnt/sd' in text or '/home/tanhe/' in text:
            absolute_paths.append(relative)
        if path.suffix == '.py':
            checked['python'] += 1
            try:
                ast.parse(text, filename=relative)
            except SyntaxError as error:
                errors.append({'path': relative, 'error': str(error)})
        else:
            checked['shell'] += 1
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append({'path': relative, 'error': result.stderr})
    report = {'scope': 'current Python/Shell source; static validation only',
              'checked': checked, 'errors': errors, 'absolute_path_files': absolute_paths,
              'source_files': rows, 'lines': sum(r['lines'] for r in rows)}
    (ROOT / 'docs/CURRENT_SOURCE_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'checked': checked, 'errors': errors, 'lines': report['lines'],
                      'absolute_path_files': len(absolute_paths)}))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
