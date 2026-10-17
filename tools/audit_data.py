"""Audit actual AgriField files; quarantine uncertain origins and duplicate components.

No image labels are read. Perceptual comparisons are screening, not proof of
identity. A component touching PhenoBench is excluded in full. Other duplicate
components retain one deterministic representative, preferring validation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image

HF_REVISION = '7f530c67572b6a572467adab840cdbbd7029b112'
ARCHIVE_SHA256 = '63eec0b4f652340feddf9011c2ee26e2df341096018bdf7f5756cc29674e8976'
INAT_METADATA_SHA256 = '8808f91f0be13832876d6e237a3b7cd870dd97a29b725b84bbb5e6ac9b9667a3'
CARD_URL = f'https://huggingface.co/datasets/dtu-pcas/AgriField-40K/raw/{HF_REVISION}/README.md'
INAT_METADATA_PATH = Path('data/iNatWeeds_metadata.json')
FIELDS = ['sample_id','path','source_prefix','source','candidate_source','source_confirmed',
          'original_id','split','original_split','width','height','mode','format','bytes',
          'sha256','pixel_sha256','phash64','dhash64','decode_ok','license','compilation_license',
          'source_url','source_revision','attribution','observation_url','status','reason','component']
# DCT-II basis; only low spatial frequencies are needed for a 64-bit pHash.
_DCT = np.cos(np.pi * (2*np.arange(32)[None,:]+1) * np.arange(8)[:,None] / 64)
_DCT[0] *= 1 / np.sqrt(2)

def image_hashes(im):
    gray = np.asarray(im.convert('L').resize((32,32), Image.Resampling.LANCZOS), dtype=np.float64)
    dct = _DCT @ gray @ _DCT.T
    values = dct.ravel()
    bits = values > np.median(values[1:])
    # Exclude DC from comparisons by pinning it to zero.
    bits[0] = False
    ph = int.from_bytes(np.packbits(bits).tobytes(), 'big')
    small = np.asarray(im.convert('L').resize((9,8), Image.Resampling.LANCZOS), dtype=np.int16)
    dh = int.from_bytes(np.packbits(small[:,1:] > small[:,:-1]).tobytes(), 'big')
    return ph, dh

class BKTree:
    def __init__(self): self.root = None
    def add(self, key, index):
        if self.root is None:
            self.root = [key, [index], {}]; return
        node = self.root
        while True:
            distance = (key ^ node[0]).bit_count()
            if distance == 0: node[1].append(index); return
            if distance not in node[2]: node[2][distance] = [key,[index],{}]; return
            node = node[2][distance]
    def query(self, key, radius):
        if self.root is None: return
        stack = [self.root]
        while stack:
            node = stack.pop(); distance = (key ^ node[0]).bit_count()
            if distance <= radius:
                for index in node[1]: yield index, distance
            stack.extend(child for edge,child in node[2].items()
                         if distance-radius <= edge <= distance+radius)

class Components:
    def __init__(self,n): self.parent=list(range(n))
    def find(self,x):
        while self.parent[x] != x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)

def write_csv(path, rows, fields=FIELDS):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('/data/datasets/AgriField-40K'))
    ap.add_argument('--archive',type=Path)
    ap.add_argument('--extract',action='store_true')
    ap.add_argument('--verify-archive',action='store_true')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--phash-radius',type=int,default=4)
    args=ap.parse_args(); started=time.time()
    root=args.root.resolve();archive=args.archive or root/'AgriField-40K.zip'
    registry=json.loads(Path('data/source_registry.json').read_text())['sources']
    inat_path=INAT_METADATA_PATH
    if hashlib.sha256(inat_path.read_bytes()).hexdigest() != INAT_METADATA_SHA256:
        raise ValueError(f'iNat metadata checksum mismatch: {inat_path}')
    inat=json.loads(inat_path.read_text())
    inat_ids={str(x['photo_id']):x for x in inat}
    archive_hash = None
    if args.verify_archive:
        print('Hashing archive...',flush=True)
        with archive.open('rb') as f: archive_hash=hashlib.file_digest(f,'sha256').hexdigest()
        if archive_hash != ARCHIVE_SHA256: raise ValueError(f'Archive checksum mismatch: {archive_hash}')
    if args.extract:
        print('Extracting archive with ZIP CRC checks...',flush=True)
        with ZipFile(archive) as z:
            members=z.infolist()
            for m in members:
                destination=(root/m.filename).resolve()
                if not destination.is_relative_to(root): raise ValueError(f'Unsafe archive path: {m.filename}')
                if m.is_dir(): continue
                # ZipExtFile checks CRC when read fully; overwrite only this dataset's files.
                destination.parent.mkdir(parents=True,exist_ok=True)
                with z.open(m) as source, destination.open('wb') as output:
                    while chunk := source.read(1024*1024): output.write(chunk)
    files=sorted(p for p in (root/'agrifield40k').rglob('*') if p.is_file())
    if not files: raise RuntimeError('No actual images found')
    prefix_counts=Counter(p.stem.partition('_')[0] for p in files)
    registry_counts={
        spec['prefix_regex']:sum(
            count for prefix,count in prefix_counts.items()
            if re.fullmatch(spec['prefix_regex'],prefix))
        for spec in registry
    }
    def audit_file(p):
        prefix,sep,original=p.stem.partition('_')
        matches=[s for s in registry if re.fullmatch(s['prefix_regex'],prefix)]
        match=matches[0] if len(matches)==1 else None
        count_matches=bool(match and registry_counts[match['prefix_regex']]==match['expected_count'])
        confirmed=bool(sep and match and match['confirmed'] and count_matches)
        unresolved_reason='unconfirmed_source_mapping'
        if match and match['confirmed'] and not count_matches:
            unresolved_reason='source_count_mismatch'
        if confirmed and match.get('original_regex'):
            if not re.fullmatch(match['original_regex'],original):
                confirmed=False
                unresolved_reason='original_id_format_unmatched'
        attribution='';observation=''
        image_license=match['license'] if confirmed else 'pending_audit'
        if prefix=='inat':
            meta=inat_ids.get(original.rsplit('_',1)[-1])
            if (not meta or Path(meta.get('file','')).stem!=original
                    or meta.get('license') not in ('cc-by','cc0')):
                confirmed=False
                unresolved_reason='inat_metadata_unresolved'
            else:
                attribution=meta.get('attribution',''); observation=meta.get('observation_url','')
                image_license={'cc-by':'CC-BY-4.0','cc0':'CC0-1.0'}[meta['license']]
        source=match['source'] if confirmed else 'pending_audit'
        reason='phenobench_source' if source=='PhenoBench' else ('' if confirmed else unresolved_reason)
        row=dict(sample_id=p.stem,path=str(p.relative_to(root)),source_prefix=prefix,
            source=source,candidate_source=(match or {}).get('candidate_source') or
                ((match or {}).get('source','') if not confirmed else ''),
            source_confirmed=str(confirmed).lower(),original_id=original,split=p.parent.name,
            original_split=p.parent.name,width='',height='',mode='',format='',bytes=p.stat().st_size,
            sha256='',pixel_sha256='',phash64='',dhash64='',decode_ok='false',
            license=image_license if confirmed else 'pending_audit',compilation_license='CC-BY-SA-4.0',
            source_url=CARD_URL,source_revision=HF_REVISION,attribution=attribution,observation_url=observation,
            status='exclude' if source=='PhenoBench' else ('include' if confirmed else 'pending_audit'),
            reason=reason,component='')
        raw=p.read_bytes();row['sha256']=hashlib.sha256(raw).hexdigest()
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.load();row.update(width=im.width,height=im.height,mode=im.mode,format=im.format)
                rgb=im.convert('RGB')
                row['pixel_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
                ph,dh=image_hashes(rgb);row.update(phash64=f'{ph:016x}',dhash64=f'{dh:016x}',decode_ok='true')
        except Exception as exc:
            row['status']='exclude';row['reason']='decode_error:'+str(exc)
        return row
    print(f'Auditing {len(files)} actual images...',flush=True)
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,row in enumerate(pool.map(audit_file,files),1):
            rows.append(row)
            if i%2000==0: print(f'decoded {i}/{len(files)}',flush=True)
    components=Components(len(rows));edges=[];by_sha={};by_pixel={};tree=BKTree()
    for i,row in enumerate(rows):
        if row['decode_ok']!='true': continue
        j=by_sha.get(row['sha256']);kind='sha256'
        if j is None: j=by_pixel.get(row['pixel_sha256']);kind='pixel_sha256'
        if j is not None:
            components.union(i,j);edges.append((j,i,kind,0,0))
        by_sha.setdefault(row['sha256'],i);by_pixel.setdefault(row['pixel_sha256'],i)
        ph=int(row['phash64'],16);dh=int(row['dhash64'],16)
        for j,distance in tree.query(ph,args.phash_radius):
            other=rows[j];dhd=(dh ^ int(other['dhash64'],16)).bit_count()
            # Intersection of two screeners reduces unrelated smooth-scene collisions.
            if dhd<=6 and row['pixel_sha256']!=other['pixel_sha256']:
                components.union(i,j);edges.append((j,i,'perceptual_candidate',distance,dhd))
        tree.add(ph,i)
        if (i+1)%5000==0: print(f'duplicate screening {i+1}/{len(rows)}',flush=True)
    groups=defaultdict(list)
    for i in range(len(rows)): groups[components.find(i)].append(i)
    for indices in groups.values():
        component=rows[min(indices)]['sample_id']
        for i in indices: rows[i]['component']=component
        pheno=any(rows[i]['source']=='PhenoBench' for i in indices)
        included=[i for i in indices if rows[i]['status']=='include']
        keeper=min(included,key=lambda i:(rows[i]['split']!='val',rows[i]['sample_id'])) if included else None
        for i in indices:
            if pheno and rows[i]['source']!='PhenoBench':
                rows[i]['status']='exclude';rows[i]['reason']='duplicate_component_touches_phenobench'
            elif not pheno and rows[i]['status']=='include' and i!=keeper:
                rows[i]['status']='exclude';rows[i]['reason']='duplicate_component_nonrepresentative'
    if len({r['sample_id'] for r in rows})!=len(rows): raise RuntimeError('Nonunique sample IDs')
    write_csv('data/all_samples.csv',rows)
    # The training manifests are intentionally include-only.  The complete
    # decision trail remains in all_samples.csv and exclusion_ledger.csv.
    write_csv('pretrain_manifest.csv',[r for r in rows if r['split']=='train' and r['status']=='include'])
    write_csv('val_manifest.csv',[r for r in rows if r['split']=='val' and r['status']=='include'])
    write_csv('data/exclusion_ledger.csv',[r for r in rows if r['status']!='include'])
    edge_rows=[dict(sample_a=rows[a]['sample_id'],sample_b=rows[b]['sample_id'],kind=k,
                    phash_distance=pd,dhash_distance=dd,cross_split=rows[a]['split']!=rows[b]['split'])
               for a,b,k,pd,dd in edges]
    write_csv('data/duplicate_pairs.csv',edge_rows,['sample_a','sample_b','kind','phash_distance','dhash_distance','cross_split'])
    by_source=defaultdict(Counter)
    for r in rows: by_source[r['source']][r['split']+'_'+r['status']]+=1
    summary=dict(images=len(rows),archive_size=archive.stat().st_size,archive_sha256=archive_hash,
        expected_archive_sha256=ARCHIVE_SHA256,revision=HF_REVISION,
        source_registry_match_counts=registry_counts,
        inat_metadata_path=str(inat_path),inat_metadata_sha256=INAT_METADATA_SHA256,
        split_counts=dict(Counter(r['split'] for r in rows)),
        source_prefix_counts=dict(Counter(r['source_prefix'] for r in rows)),
        source_counts={k:dict(v) for k,v in sorted(by_source.items())},
        status_counts=dict(Counter(r['status'] for r in rows)),
        reason_counts=dict(Counter(r['reason'] for r in rows if r['reason'])),
        formats=dict(Counter(r['format'] for r in rows)),
        dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
        modes=dict(Counter(r['mode'] for r in rows)),decode_errors=sum(r['decode_ok']!='true' for r in rows),
        duplicate_edges=dict(Counter(e[2] for e in edges)),
        duplicate_components=sum(len(v)>1 for v in groups.values()),
        cross_split_edges=sum(rows[a]['split']!=rows[b]['split'] for a,b,*_ in edges),
        phash_radius=args.phash_radius,dhash_radius=6,seconds=time.time()-started)
    included=[r for r in rows if r['status']=='include']
    train_components={r['component'] for r in included if r['split']=='train'}
    val_components={r['component'] for r in included if r['split']=='val'}
    assert not train_components & val_components
    assert not any(r['source']=='PhenoBench' or r['source_confirmed']!='true' for r in included)
    summary['included_split_counts']=dict(Counter(r['split'] for r in included))
    summary['manifest_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in map(Path,['pretrain_manifest.csv','val_manifest.csv','data/all_samples.csv'])}
    Path('reports/data_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
