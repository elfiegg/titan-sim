"""Read-only prepare-runtime handoff checks; no environment repair or job launch.

The upstream version-1 artifacts are agent-written, extensible objects. This
module checks evidence needed by simulation without pretending to certify the
currently running container. Caller-supplied expected identities bind a request
to the qualified scope; changes require new capability evidence.
"""
import hashlib
import json
from pathlib import Path
import re


def _read(path):
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('runtime artifact must be a JSON object: '+str(path))
    return value, hashlib.sha256(raw).hexdigest()


def image_identity(image):
    if not isinstance(image, dict):
        return None
    for key in ('digest', 'sha256', 'file_hash'):
        value = image.get(key)
        if isinstance(value, str) and re.fullmatch(r'(sha256:)?[a-fA-F0-9]{64}', value):
            return 'sha256:'+value.removeprefix('sha256:').lower()
    return None


def check_runtime(manifest_path, capability_report=None, expected=None):
    """Return readiness and immutable handoff fingerprints, without executing code.

    expected optionally specifies codebase_commit, image_identity, recipe (full
    object), target (subset of explicit fields), and required_checks (check IDs).
    Local evidence paths are checked relative to their report directory. Remote
    URIs remain unverified and therefore blocked until staged by the caller.
    """
    manifest_path = Path(manifest_path)
    manifest, manifest_hash = _read(manifest_path)
    expected = expected or {}
    if not isinstance(expected, dict):
        raise ValueError('expected runtime identity must be an object')
    gaps = []
    for field in ('codebase', 'recipe', 'target', 'validation'):
        if not isinstance(manifest.get(field), dict):
            raise ValueError('runtime '+field+' must be an object')
    if manifest.get('schema_version') != 1:
        gaps.append('Unsupported or missing runtime schema_version; expected 1.')
    if manifest.get('status') != 'validated':
        gaps.append('Runtime status is not validated: '+str(manifest.get('status')))
    identity = image_identity(manifest.get('final_image'))
    if identity is None:
        gaps.append('Final image lacks an immutable SHA256 identity.')
    commit = manifest.get('codebase', {}).get('resolved_commit')
    if not isinstance(commit, str) or not re.fullmatch(r'[a-fA-F0-9]{40,64}', commit):
        gaps.append('Codebase lacks a full resolved commit.')
    for field in ('run_id', 'recipe', 'target', 'reproduce'):
        if not manifest.get(field):
            gaps.append('Missing runtime handoff field: '+field)
    if expected.get('codebase_commit') is not None and expected['codebase_commit'] != commit:
        gaps.append('Requested codebase commit differs from the prepared runtime.')
    if expected.get('image_identity') is not None and expected['image_identity'] != identity:
        gaps.append('Requested image differs from the prepared runtime.')
    if expected.get('recipe') is not None and expected['recipe'] != manifest.get('recipe'):
        gaps.append('Requested recipe/overrides differ from the qualified recipe; requalify changed features.')
    target = expected.get('target', {})
    if not isinstance(target, dict):
        raise ValueError('expected target must be an object')
    for key, value in target.items():
        if manifest.get('target', {}).get(key) != value:
            gaps.append('Requested target differs from qualification: '+key)
    report_ref = capability_report or manifest.get('validation', {}).get('capability_report')
    if isinstance(report_ref, dict):
        report_ref = report_ref.get('path')
    report, report_hash, report_path = {}, None, None
    if not isinstance(report_ref, (str, Path)) or '://' in str(report_ref):
        gaps.append('Capability report must be staged as a readable local artifact.')
    else:
        report_path = Path(report_ref)
        if not report_path.is_absolute():
            report_path = manifest_path.parent / report_path
        try:
            report, report_hash = _read(report_path)
        except (OSError, ValueError) as error:
            gaps.append('Cannot read capability report: '+str(error))
    checks = report.get('checks', [])
    if report.get('schema_version') != 1:
        gaps.append('Unsupported or missing capability report schema_version; expected 1.')
    if not manifest.get('run_id') or report.get('run_id') != manifest.get('run_id'):
        gaps.append('Capability report run_id differs from the runtime handoff.')
    for field in ('codebase', 'recipe'):
        if field in report and report[field] != manifest.get(field):
            gaps.append('Capability report '+field+' differs from the runtime handoff.')
    if not isinstance(checks, list):
        raise ValueError('capability report checks must be a list')
    by_id = {}
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get('id'), str):
            raise ValueError('capability check requires a string id')
        if check['id'] in by_id:
            raise ValueError('duplicate capability check id: '+check['id'])
        by_id[check['id']] = check
    requested = expected.get('required_checks', [])
    if not isinstance(requested, list) or not all(isinstance(x, str) for x in requested):
        raise ValueError('required_checks must be a list of check IDs')
    declared = manifest.get('validation', {}).get('required_checks', [])
    if not isinstance(declared, list) or not all(isinstance(x, str) for x in declared):
        raise ValueError('validation.required_checks must be a list of check IDs')
    required = set(requested) | set(declared) | {key for key, c in by_id.items() if c.get('required') is True}
    evidence_hashes = {}
    if not required:
        gaps.append('No required capability checks are recorded.')
    for key in sorted(required):
        check = by_id.get(key)
        if check is None:
            gaps.append('Required capability check is missing: '+key)
            continue
        if check.get('status') != 'pass' or check.get('exit_code') != 0 or isinstance(check.get('exit_code'), bool):
            gaps.append('Required capability has no successful execution: '+key)
        if not check.get('command') or not check.get('assertions'):
            gaps.append('Capability lacks command/assertion evidence: '+key)
        candidate = check.get('image_identity')
        candidate_id = image_identity(candidate) if isinstance(candidate, dict) else candidate
        if candidate_id != identity:
            gaps.append('Capability image identity differs or is missing: '+key)
        if check.get('target') != manifest.get('target'):
            gaps.append('Capability target differs or is missing: '+key)
        paths = check.get('evidence_paths')
        if not isinstance(paths, list) or not paths:
            gaps.append('Capability has no evidence paths: '+key)
            continue
        for evidence in paths:
            if not isinstance(evidence, str) or '://' in evidence:
                gaps.append('Capability evidence must be staged for verification: '+key)
            elif report_path is None or not (report_path.parent / evidence).is_file():
                gaps.append('Capability evidence is missing: '+key+': '+evidence)
            else:
                evidence_hashes[evidence] = hashlib.sha256((report_path.parent / evidence).read_bytes()).hexdigest()
    return dict(schema_version=1, status='blocked' if gaps else 'ready',
                scope='Artifact consistency only; not a probe of the running container or full-scale qualification.',
                run_id=manifest.get('run_id'), image_identity=identity,
                codebase=manifest.get('codebase'), recipe=manifest.get('recipe'),
                target=manifest.get('target'), validation=manifest.get('validation'),
                runtime_manifest_sha256=manifest_hash, capability_report_sha256=report_hash,
                evidence_sha256=evidence_hashes,
                expected=expected, required_checks=sorted(required), unresolved=gaps)
