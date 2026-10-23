#!/usr/bin/env python
"""Check that every registered download URL actually resolves.

For each unique non-empty ``download_url`` in the catalog we issue a bounded
HTTP request and record the status.  Results are written as JSON so the
registry builder can attach real evidence instead of guessing.

A 403/429 is recorded as ``reachable_but_blocked`` (the host exists and refuses
automated access) and never upgraded to "verified".  A DNS/connection error is
recorded as ``unreachable``.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.registry.catalog import all_sources

USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/122.0 Safari/537.36')


def check(url: str, timeout: float = 20.0) -> Dict[str, object]:
    request = urllib.request.Request(url, method='GET',
                                     headers={'User-Agent': USER_AGENT,
                                              'Range': 'bytes=0-1023'})
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
            status = resp.status
            final = resp.geturl()
        if 200 <= status < 400:
            outcome = 'ok'
        elif status in (401, 403, 429):
            outcome = 'reachable_but_blocked'
        else:
            outcome = f'http_{status}'
        return {'url': url, 'status': status, 'final_url': final,
                'outcome': outcome}
    except urllib.error.HTTPError as exc:
        outcome = ('reachable_but_blocked' if exc.code in (401, 403, 429)
                   else f'http_{exc.code}')
        return {'url': url, 'status': exc.code, 'final_url': exc.geturl(),
                'outcome': outcome}
    except Exception as exc:  # noqa: BLE001 - record any failure verbatim
        return {'url': url, 'status': None, 'final_url': None,
                'outcome': 'unreachable', 'error': f'{type(exc).__name__}: {exc}'}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='reports/stage5a_url_check.json')
    parser.add_argument('--timeout', type=float, default=20.0)
    args = parser.parse_args()

    urls = sorted({str(r['download_url']) for r in all_sources()
                   if str(r['download_url']).startswith('http')})
    results = {}
    for i, url in enumerate(urls, 1):
        results[url] = check(url, args.timeout)
        print(f'[{i}/{len(urls)}] {results[url]["outcome"]:<24} {url}', flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    ok = sum(1 for r in results.values() if r['outcome'] == 'ok')
    blocked = sum(1 for r in results.values()
                  if r['outcome'] == 'reachable_but_blocked')
    print(f'\n{ok}/{len(results)} ok, {blocked} reachable-but-blocked, '
          f'{len(results) - ok - blocked} failed -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
