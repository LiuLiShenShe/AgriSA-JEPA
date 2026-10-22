#!/usr/bin/env python
"""Verify the arXiv references cited in the Stage 4A method-design report."""
import re
import sys
import urllib.request

IDS = {
    'I-JEPA (Assran et al.)': '2301.08243',
    'AttMask (Kakogeorgiou et al.)': '2203.12719',
    'Semantic guidance for SSL (Lebedev?)': '1905.01229',
    'SemSeg-aware masking (Weinzaepfel?)': '2207.12406',
}


def query(arxiv_id):
    url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode('utf-8')


def main():
    for label, arxiv_id in IDS.items():
        try:
            text = query(arxiv_id)
        except Exception as exc:                       # noqa: BLE001
            print(f'{label}: FETCH FAILED {exc}')
            continue
        entry = re.search(r'<entry>.*?</entry>', text, re.S)
        if not entry:
            print(f'{label}: NO ENTRY')
            continue
        body = entry.group(0)
        title = re.search(r'<title>(.*?)</title>', body, re.S)
        authors = re.findall(r'<name>(.*?)</name>', body)
        summary = re.search(r'<summary>(.*?)</summary>', body, re.S)
        print(f'{label}')
        print(f'  id     : {arxiv_id}')
        print(f'  title  : {re.sub(chr(10), " ", title.group(1)).strip()}')
        print(f'  authors: {", ".join(authors)}')
        print(f'  abstract: {re.sub(chr(10), " ", summary.group(1)).strip()[:700]}')
        print()


if __name__ == '__main__':
    sys.exit(main())
