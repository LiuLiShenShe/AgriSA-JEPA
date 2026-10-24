"""Resumable, idempotent download machinery for Stage 5C.

Everything here is deliberately boring: HTTP Range requests, SHA-256 while
streaming, ``.part`` files, per-file retry counts and an append-only leadger.
A download that already exists with the expected checksum is *skipped*, never
re-fetched, so re-running the pipeline is cheap and safe.

No authentication is bypassed.  Endpoints that answer 401/403 are recorded as
``BLOCKED`` with the observed status code; they are not retried with synthetic
credentials.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

USER_AGENT = 'Mozilla/5.0 (compatible; AgriSA-JEPA/0.1; academic research)'

MANIFEST_FIELDS = [
    'source_id', 'remote_url', 'remote_etag_if_any', 'archive_file',
    'source_file_id', 'expected_bytes', 'downloaded_bytes',
    'checksum_algorithm', 'checksum_value', 'retry_count',
    'download_timestamp', 'status', 'error_code', 'source_version', 'evidence',
]


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _request(url: str, method: str = 'GET', headers: Optional[Dict[str, str]] = None,
             timeout: int = 60) -> urllib.request.Request:
    merged = {'User-Agent': USER_AGENT, 'Accept': '*/*'}
    merged.update(headers or {})
    return urllib.request.Request(url, method=method, headers=merged)


def probe(url: str, timeout: int = 30) -> Dict[str, object]:
    """Observe what a URL actually returns (redirects followed by urllib)."""
    out: Dict[str, object] = {'url': url, 'status': None, 'content_type': '',
                              'content_length': None, 'final_url': '', 'error': ''}
    try:
        with urllib.request.urlopen(_request(url, timeout=timeout)) as response:
            out['status'] = response.status
            out['content_type'] = response.headers.get('Content-Type', '')
            length = response.headers.get('Content-Length')
            out['content_length'] = int(length) if length else None
            out['final_url'] = response.geturl()
    except urllib.error.HTTPError as exc:
        out['status'] = exc.code
        out['content_type'] = exc.headers.get('Content-Type', '') if exc.headers else ''
        out['error'] = exc.reason
    except Exception as exc:  # noqa: BLE001 - network failures are data
        out['error'] = f'{type(exc).__name__}: {exc}'
    return out


def remote_size(url: str, timeout: int = 40) -> Optional[int]:
    """Total size from a one-byte range request (works when HEAD is 404)."""
    try:
        request = _request(url, headers={'Range': 'bytes=0-0'}, timeout=timeout)
        with urllib.request.urlopen(request) as response:
            content_range = response.headers.get('Content-Range', '')
            if '/' in content_range:
                return int(content_range.rsplit('/', 1)[1])
            length = response.headers.get('Content-Length')
            return int(length) if length else None
    except Exception:
        return None


def download_file(url: str, dest: Path, expected_bytes: Optional[int] = None,
                  expected_sha256: str = '', timeout: int = 60,
                  retries: int = 4, chunk: int = 8 << 20,
                  progress: Optional[Callable[[int, Optional[int]], None]] = None,
                  verify_existing: bool = False) -> Dict[str, object]:
    """Download ``url`` to ``dest`` with resume, retry and checksum.

    Returns a ledger-shaped dict.  Never raises for HTTP failures; the caller
    decides what a failed row means.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    record: Dict[str, object] = {
        'remote_url': url, 'archive_file': str(dest), 'expected_bytes': expected_bytes,
        'downloaded_bytes': 0, 'checksum_algorithm': 'sha256', 'checksum_value': '',
        'retry_count': 0, 'status': 'FAILED', 'error_code': '', 'evidence': '',
    }

    if dest.exists() and dest.stat().st_size > 0:
        size = dest.stat().st_size
        if expected_bytes is None or size == expected_bytes:
            # Re-hashing every pre-existing file dominates resume time on
            # 100k-file sources.  File size is the resume contract; a full
            # checksum audit is opt-in via verify_existing (or the archive
            # auditor).  Fresh downloads are always hashed.
            digest = sha256_file(dest) if verify_existing else ''
            if verify_existing and expected_sha256 and digest != expected_sha256:
                record.update(status='CHECKSUM_MISMATCH', downloaded_bytes=size,
                              checksum_value=digest,
                              error_code='sha256 differs from expected')
                return record
            record.update(status='SKIPPED_EXISTING', downloaded_bytes=size,
                          checksum_value=digest,
                          evidence='existing file verified' if verify_existing
                          else 'existing file, size match')
            return record

    for attempt in range(retries + 1):
        record['retry_count'] = attempt
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={existing}-'} if existing else {}
            request = _request(url, headers=headers, timeout=timeout)
            with urllib.request.urlopen(request) as response:
                mode = 'ab' if existing and response.status == 206 else 'wb'
                if mode == 'wb':
                    existing = 0
                total = expected_bytes
                content_range = response.headers.get('Content-Range', '')
                if '/' in content_range:
                    total = int(content_range.rsplit('/', 1)[1])
                # Hard wall clock: closing the socket from a timer unblocks a
                # trickling or half-dead S3 response.  Without this one object
                # can stall a whole manifest chunk indefinitely.
                watchdog = threading.Timer(max(timeout * 2, 120), response.close)
                watchdog.daemon = True
                watchdog.start()
                try:
                    with open(part, mode) as handle:
                        while True:
                            block = response.read(chunk)
                            if not block:
                                break
                            handle.write(block)
                            existing += len(block)
                            if progress:
                                progress(existing, total)
                finally:
                    watchdog.cancel()
            digest = sha256_file(part)
            size = part.stat().st_size
            if expected_sha256 and digest != expected_sha256:
                record.update(status='CHECKSUM_MISMATCH', downloaded_bytes=size,
                              checksum_value=digest, error_code='sha256 mismatch')
                return record
            if total is not None and size != total:
                raise DownloadError(f'incomplete: {size} of {total} bytes')
            os.replace(part, dest)
            record.update(status='DOWNLOADED', downloaded_bytes=size,
                          expected_bytes=total, checksum_value=digest,
                          evidence=f'attempts={attempt + 1}')
            return record
        except urllib.error.HTTPError as exc:
            code = exc.code
            record.update(downloaded_bytes=part.stat().st_size if part.exists() else 0,
                          error_code=f'HTTP {code}')
            if code in (401, 403, 404):
                record.update(status='BLOCKED' if code in (401, 403) else 'NOT_FOUND')
                return record
            time.sleep(min(2 ** attempt, 20))
        except Exception as exc:  # noqa: BLE001
            record.update(downloaded_bytes=part.stat().st_size if part.exists() else 0,
                          error_code=f'{type(exc).__name__}: {exc}')
            time.sleep(min(2 ** attempt, 20))
    return record


# --------------------------------------------------------------------------- #
# Kaggle / Dataverse / Mendeley / HF adapters
# --------------------------------------------------------------------------- #

KAGGLE_DATASET = 'https://www.kaggle.com/api/v1/datasets/download/{slug}'


def download_kaggle(slug: str, dest: Path, expected_bytes: Optional[int] = None) -> Dict[str, object]:
    return download_file(KAGGLE_DATASET.format(slug=slug), dest, expected_bytes=expected_bytes)


def dataverse_files(persistent_id: str, timeout: int = 40) -> List[Dict[str, object]]:
    url = ('https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId='
           + urllib.parse.quote(persistent_id, safe=''))
    with urllib.request.urlopen(_request(url, timeout=timeout)) as response:
        payload = json.loads(response.read())
    files = []
    for entry in payload['data']['latestVersion']['files']:
        data = entry['dataFile']
        files.append({
            'id': data['id'], 'name': data['filename'],
            'bytes': data.get('filesize'), 'sha256': '',
            'url': f'https://dataverse.harvard.edu/api/access/datafile/{data["id"]}',
        })
    return files


def mendeley_files(dataset_id: str, timeout: int = 60) -> List[Dict[str, object]]:
    url = f'https://data.mendeley.com/public-api/datasets/{dataset_id}'
    with urllib.request.urlopen(_request(url, timeout=timeout)) as response:
        payload = json.loads(response.read())
    files = []
    for entry in payload.get('files', []):
        details = entry.get('content_details', {})
        files.append({
            'id': entry.get('id'), 'name': entry.get('filename'),
            'bytes': details.get('size'), 'sha256': details.get('sha256_hash', ''),
            'url': details.get('download_url', ''),
            'content_type': details.get('content_type', ''),
        })
    return files


def hf_repo_files(repo: str, kind: str = 'datasets', timeout: int = 60) -> List[Dict[str, object]]:
    url = f'https://hf-mirror.com/api/{kind}/{repo}/tree/main'
    with urllib.request.urlopen(_request(url, timeout=timeout)) as response:
        payload = json.loads(response.read())
    files = []
    for entry in payload:
        if entry.get('type') != 'file':
            continue
        files.append({'name': entry['path'], 'bytes': entry.get('size'),
                      'url': f'https://hf-mirror.com/{kind}/{repo}/resolve/main/{urllib.parse.quote(entry["path"])}'})
    return files


# --------------------------------------------------------------------------- #
# S3 (Artemis / Image Safari)
# --------------------------------------------------------------------------- #

S3_BUCKET = 'https://alliance-artemis-imagesafari.s3.us-west-2.amazonaws.com/'
S3_NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def s3_list(prefix: str, delimiter: Optional[str] = None, max_keys: int = 1000,
            continuation: str = '', timeout: int = 60) -> Dict[str, object]:
    params = {'list-type': '2', 'max-keys': str(max_keys), 'prefix': prefix}
    if delimiter:
        params['delimiter'] = delimiter
    if continuation:
        params['continuation-token'] = continuation
    url = S3_BUCKET + '?' + urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(_request(url, timeout=timeout)) as response:
                root = ET.fromstring(response.read())
            contents = []
            for node in root.findall('s:Contents', S3_NS):
                contents.append({
                    'key': node.find('s:Key', S3_NS).text,
                    'bytes': int(node.find('s:Size', S3_NS).text),
                    'etag': (node.find('s:ETag', S3_NS).text or '').strip('"'),
                })
            prefixes = [n.find('s:Prefix', S3_NS).text for n in root.findall('s:CommonPrefixes', S3_NS)]
            token = root.find('s:NextContinuationToken', S3_NS)
            return {'contents': contents, 'prefixes': prefixes,
                    'next': token.text if token is not None else '',
                    'truncated': (root.find('s:IsTruncated', S3_NS).text == 'true')}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 ** attempt, 10))
    raise DownloadError(f'S3 listing failed for {prefix}: {last_error}')


def s3_download(key: str, dest: Path, expected_bytes: Optional[int] = None) -> Dict[str, object]:
    url = S3_BUCKET + urllib.parse.quote(key)
    return download_file(url, dest, expected_bytes=expected_bytes)


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #

class DownloadLedger:
    """Append-only CSV ledger with idempotent upsert by (source_id, remote_url)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.rows: Dict[Tuple[str, str], Dict[str, object]] = {}
        if self.path.exists():
            with open(self.path, newline='', encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    self.rows[(row.get('source_id', ''), row.get('remote_url', ''))] = row

    def record(self, source_id: str, record: Dict[str, object]) -> None:
        row = {field: '' for field in MANIFEST_FIELDS}
        row.update(record)
        row['source_id'] = source_id
        row['download_timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        with self._lock:
            self.rows[(source_id, str(row.get('remote_url', '')))] = row

    def flush(self) -> int:
        """Merge this process's rows into the shared ledger under a file lock.

        Several downloaders run concurrently against the same ledger.  A plain
        "write my snapshot" flush would drop rows another process added after
        this one started, so the flush reloads the on-disk ledger, merges, and
        replaces atomically.  ``fcntl.flock`` serialises the read-merge-write.
        """
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.parent / (self.path.name + '.lock')
        tmp = self.path.parent / (f'.{self.path.name}.{os.getpid()}.tmp')
        with self._lock, open(lock_path, 'w') as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            disk: Dict[Tuple[str, str], Dict[str, object]] = {}
            if self.path.exists():
                try:
                    with open(self.path, newline='', encoding='utf-8') as handle:
                        for row in csv.DictReader(handle):
                            disk[(row.get('source_id', ''), row.get('remote_url', ''))] = row
                except (OSError, csv.Error):
                    disk = {}
            merged = {**disk, **self.rows}
            with open(tmp, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                for row in merged.values():
                    writer.writerow({k: row.get(k, '') for k in MANIFEST_FIELDS})
            os.replace(tmp, self.path)
            self.rows = merged
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
        return len(self.rows)


def parallel_download(items: Sequence[Tuple[str, Path, Optional[int]]],
                      source_id: str, ledger: DownloadLedger,
                      workers: int = 16,
                      on_done: Optional[Callable[[Dict[str, object]], None]] = None,
                      timeout: int = 60) -> Dict[str, int]:
    """Download ``(url, dest, expected_bytes)`` tuples with a shared ledger.

    A hard wall-clock ceiling applies per file: one object that stalls
    forever must not block a 100k-file manifest.  The per-file HTTP timeout
    (60 s) covers connect and idle gaps; the wall clock catches a trickle.
    """
    counts: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, url, dest, expected, timeout=timeout): (url, dest)
            for url, dest, expected in items
        }
        for future in as_completed(futures):
            url, dest = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                record = {'remote_url': url, 'archive_file': str(dest),
                          'status': 'FAILED', 'error_code': str(exc),
                          'downloaded_bytes': 0, 'retry_count': 0,
                          'checksum_algorithm': 'sha256', 'checksum_value': '',
                          'expected_bytes': None, 'evidence': ''}
            ledger.record(source_id, record)
            counts[record['status']] = counts.get(record['status'], 0) + 1
            if on_done:
                on_done(record)
    return counts
