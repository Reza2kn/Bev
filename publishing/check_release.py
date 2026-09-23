"""Validate the curated release tree without running GPU inference."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote


def main():
    root = Path(__file__).resolve().parents[1]
    try:
        names = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
    except subprocess.CalledProcessError:
        names = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                 and not any(x in p.parts for x in ['.git', '__pycache__', '.pytest_cache', 'build', 'dist'])
                 and '.egg-info' not in str(p)]
    assert names, 'No release files found'
    forbidden = ('artifacts/', 'benchmarks/', 'research/', 'release-work/', '.env', 'models/', 'runtime/')
    issues = []
    secret = re.compile(r'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
    private_path = re.compile(r'/(?:home|Users)/[A-Za-z0-9_-]+/')
    for name in names:
        if name.startswith(forbidden):
            issues.append(f'Unexpected development file: {name}')
        path = root / name
        if not path.is_file():
            issues.append(f'Missing file: {name}')
            continue
        value = path.read_text()
        if secret.search(value):
            issues.append(f'Credential-like content in {name}')
        if private_path.search(value):
            issues.append(f'Private host path in {name}')
        if path.suffix == '.md':
            for link in re.findall(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)', value):
                link = unquote(link.split('#')[0])
                if not link or '://' in link or link.startswith(('mailto:', 'data:')):
                    continue
                if not (path.parent / link).exists():
                    issues.append(f'Broken relative link in {name}: {link}')
        if path.suffix == '.json':
            json.loads(value)
    prov = json.loads((root/'evaluations/persian-v1/provenance.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, expected in prov['service']['source_sha256'].items():
        if name == '__init__.py':
            previous = (root/'bev'/name).read_text().replace('0.1.2', '0.1.1').encode()
            assert hashlib.sha256(previous).hexdigest() == expected, 'Inference init changed beyond version string'
        else:
            assert sha(root/'bev'/name) == expected, f'Inference source changed: {name}'
    assert sha(root/'evaluations/persian-v1/summary.json') == prov['original_run_files_sha256']['summary.json']
    assert sha(root/'evaluations/persian-v1/runner.py') == prov['evaluation_wrapper_sha256']
    manifest = json.loads((root/'model-manifest.json').read_text())
    assert sha(root/'patches/prism-selected-logprobs.patch') == manifest['runtime_patch_sha256']
    summary = json.loads((root/'evaluations/persian-v1/summary.json').read_text())
    assert summary['completion']['valid_answers'] == 624
    assert [summary['main'][key]['correct'] for key in ['choice', 'noul', 'score']] == [229, 152, 70]
    if issues:
        raise SystemExit('\n'.join(issues))
    print(json.dumps({'passed': True, 'release_files': len(names), 'relative_links': 'valid',
                      'benchmark_source_hashes': 'app/core/models identical; init version-only', 'private_path_and_credential_scan': 'clear'}))


if __name__ == '__main__':
    main()
