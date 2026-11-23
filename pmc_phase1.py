"""Phase 1 of a PubMed Central RAG project. Python 3.9+, standard library only.
Keep this file in PMC_RAG_Phase1 beside the extracted raw and ready data folders.
Commands: python pmc_phase1.py collect | check | demo
"""
import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'PMC_RAG_Raw_Sources'
READY = ROOT / 'PMC_RAG_Ready_data'
OUTPUT = ROOT / 'phase1_output'
REVISION = 'a661e8c14f973398380c8865cf2f27a535aaaf6d'
SOURCE = f'https://raw.githubusercontent.com/jayded/evidence-inference/{REVISION}/annotations/xml_files'
QA = 'PMC_Question_Answers_5MB.jsonl'
CHAT = 'PMC_Dialogues_5MB.jsonl'
CORPUS = 'PMC_Passages.jsonl'
EXAMPLE_IDS = ['EI-1686', 'EI-11587', 'EI-4969']


def locate_folder(folder, filename):
    """Find the data even when Windows extraction adds an extra folder level."""
    direct = folder / filename
    if direct.is_file():
        return folder
    if not folder.is_dir():
        variants = [p for p in ROOT.iterdir() if p.is_dir() and p.name.casefold() == folder.name.casefold()]
        if len(variants) != 1:
            raise FileNotFoundError(f'Place {folder.name} beside this script in {ROOT}.')
        folder = variants[0]
    matches = [p.parent for p in folder.rglob(filename) if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f'Expected one {filename} under {folder}; found {len(matches)}.')
    return matches[0]


def csv_rows(folder, filename):
    with (folder / filename).open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def records(folder, name):
    with (folder / name).open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(text):
    return ''.join(ch for ch in unicodedata.normalize('NFKD', text.casefold()) if ch.isalnum())


def collect(pmcid, all_articles):
    raw = locate_folder(RAW, 'source_manifest.csv')
    manifest = {row['pmcid']: row for row in csv_rows(raw, 'source_manifest.csv')}
    selected = sorted(manifest) if all_articles else [pmcid.upper()]
    for pmc in selected:
        ensure(re.fullmatch(r'PMC[0-9]+', pmc) and pmc in manifest, 'Choose a PMCID from source_manifest.csv.')
        collect_article(pmc, manifest[pmc])


def collect_article(pmcid, ref):
    folder = OUTPUT / 'collected'
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f'{pmcid}.nxml'
    if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == ref['sha256']:
        print(f'Already verified {pmcid}')
        return
    url = f'{SOURCE}/{pmcid}.nxml'
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    ensure(digest == ref['sha256'], 'Downloaded article differs from the recorded source.')
    dest.write_bytes(data)
    from datetime import datetime, timezone
    log = dict(collected_at_utc=datetime.now(timezone.utc).isoformat(), pmcid=pmcid, url=url,
               bytes=len(data), sha256=digest, license=ref['article_license_url'])
    with (folder/'acquisition_log.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(log)+'\n')
    print(f'Collected {pmcid} and matched its SHA-256. Saved {dest}')


def check():
    raw = locate_folder(RAW, 'source_manifest.csv')
    ready = locate_folder(READY, QA)
    manifest = csv_rows(raw, 'source_manifest.csv')
    prompts = csv_rows(raw, 'annotations/prompts_project_subset.csv')
    annotations = csv_rows(raw, 'annotations/annotations_project_subset.csv')
    qa, chats, passages = [records(ready, f) for f in (QA, CHAT, CORPUS)]
    sizes = {Path(f).name: (ready / f).stat().st_size for f in (QA, CHAT)}
    ensure(all(v == 5_000_000 for v in sizes.values()), 'Unexpected example file size.')
    source_paras = {}
    sources = {}
    for ref in manifest:
        pmc = ref['pmcid']
        xml = (raw / ref['local_xml']).read_bytes()
        ensure(hashlib.sha256(xml).hexdigest() == ref['sha256'], f'XML checksum mismatch: {pmc}')
        article = ET.fromstring(xml)
        ids = {''.join(e.itertext()).strip().removeprefix('PMC')
               for e in article.findall('./front/article-meta/article-id')
               if e.get('pub-id-type') in ('pmc','pmcid')}
        ensure(pmc[3:] in ids and pmc not in sources, f'Unexpected XML article ID: {pmc}')
        source_paras[pmc] = {' '.join(''.join(p.itertext()).split()) for p in article.iter('p')}
        sources[pmc] = ref
    corpus = {p['id']: p for p in passages}
    ensure(len(corpus) == len(passages), 'Duplicate passage ID.')
    for p in passages:
        ensure(p['text'] in source_paras[p['pmcid']], f'Passage text differs from XML: {p["id"]}')
    prompts_by_id = {p['PromptID']: p for p in prompts}
    ensure(len(prompts_by_id)==len(prompts), 'Duplicate prompt ID.')
    by_prompt = defaultdict(list)
    for a in annotations:
        ensure(a['PromptID'] in prompts_by_id and a['PMCID'] == prompts_by_id[a['PromptID']]['PMCID'],
               'Unexpected annotation ID.')
        by_prompt[a['PromptID']].append(a)
    used = set()
    for r in qa + chats:
        pid = r['id'].removeprefix('EI-')
        pmc = r['source']['pmcid']
        ensure(pid not in used and pid in prompts_by_id, f'Invalid or duplicate example: {pid}')
        used.add(pid)
        src = prompts_by_id[pid]
        ensure('PMC'+src['PMCID'] == pmc and r['source']['article_url'] == sources[pmc]['pmc_url'],
               f'Source mismatch: {pid}')
        ensure(all(r[k]==src[k.title()].strip() for k in ('intervention','comparator','outcome')),
               f'Prompt fields differ from source: {pid}')
        evidence = corpus[r['evidence_passage_id']]
        ensure(evidence['pmcid'] == pmc and any(p['id']==evidence['id'] for p in r['passages']),
               f'Incorrect evidence article or missing passage: {pid}')
        label_matches = [a for a in by_prompt[pid] if a['Valid Label']=='True'
                         and a['Valid Reasoning']=='True' and a['Label']==r['label']]
        norm_evidence = normalized(evidence['text'])
        ensure(any(span and span in norm_evidence for a in label_matches
                   for span in [normalized(a['Annotations'])]), f'Unmatched annotated evidence: {pid}')
    ensure(used==set(prompts_by_id), 'Some source prompts have no prepared example.')
    ensure(not ({r['source']['pmcid'] for r in qa} & {r['source']['pmcid'] for r in chats}),
           'Article overlap between example sets.')
    summary = dict(verified_xml_articles=len(sources), original_prompts=len(prompts),
                   original_annotation_rows=len(annotations), unique_passages=len(corpus),
                   qa_examples=len(qa), dialogues=len(chats), dataset_bytes=sizes)
    OUTPUT.mkdir(exist_ok=True)
    report = json.dumps(summary, indent=2)
    (OUTPUT/'validation.json').write_text(report+'\n', encoding='utf-8')
    print(report)


def demo():
    ready = locate_folder(READY, QA)
    examples = {r['id']: r for r in records(ready, QA) if r['id'] in EXAMPLE_IDS}
    ensure(set(examples)==set(EXAMPLE_IDS),'Missing a saved demo example.')
    esc = html.escape
    cards = []
    for rid in EXAMPLE_IDS:
        row = examples[rid]
        source = row['source']
        ev = next(p for p in row['passages'] if p['id']==row['evidence_passage_id'])
        # Expandable HTML elements work directly from a local file without scripts or network calls.
        cards.append(f"""<details><summary>{esc(row['question'])}</summary>
        <div class="body"><h2>Saved answer</h2><p>{esc(row['answer'])}</p>
        <h2>Supporting passage</h2><blockquote>{esc(ev['text'])}</blockquote>
        <small>{esc(rid)} &middot; {esc(ev['id'])} &middot; {esc(ev['section'])}</small>
        <p><a href="{esc(source['article_url'], quote=True)}" target="_blank" rel="noopener noreferrer">{esc(source['title'])} ↗</a></p>
        <small>{esc(', '.join(source['authors']))} &middot; <a href="{esc(source['license'],quote=True)}">Article license</a></small>
        </div></details>""")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PMC study explorer</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f5f7f3;color:#18382d;font:1rem/1.6 'Segoe UI',Arial,sans-serif}
    main{max-width:860px;margin:0 auto;padding:clamp(20px,5vw,65px) 20px}h1{font:500 clamp(2rem,7vw,3.4rem)/1.1 Georgia,serif;margin:0}
    h2{font-size:1rem;margin:22px 0 6px}p{margin:9px 0 18px}.intro{margin:16px 0 28px;color:#41584f}
    details{background:white;border:1px solid #bdd0c3;margin:12px 0;border-radius:8px;overflow-wrap:anywhere}
    summary,.body{padding:15px 19px}summary{cursor:pointer;font-weight:600}summary:focus-visible{outline:3px solid #468e65;outline-offset:2px}
    .body{border-top:1px solid #dfebe1}blockquote{margin:0 0 10px;padding:13px 16px;background:#eef4ed;border-left:3px solid #528267}
    small{color:#4c6156}a{color:#185741;text-underline-offset:3px}
    </style></head><body><main><h1>PMC study explorer</h1>
    <p class="intro">Offline prototype. Open a saved research question to inspect its answer, supporting passage and published source.</p>"""
    out = OUTPUT / 'ui'
    out.mkdir(parents=True,exist_ok=True)
    dest = out / 'index.html'
    dest.write_text(page + ''.join(cards) + '</main></body></html>', encoding='utf-8')
    print(f'Open {dest} in your browser.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    col = sub.add_parser('collect', help='Download one original XML and record its provenance')
    sel = col.add_mutually_exclusive_group()
    sel.add_argument('pmcid', nargs='?', default='PMC4737200')
    sel.add_argument('--all', action='store_true', help='Collect all articles, skipping verified local copies')
    sub.add_parser('check', help='Validate both extracted datasets offline')
    sub.add_parser('demo', help='Build a standalone HTML demo using three real records')
    args = parser.parse_args()
    try:
        if args.command == 'collect':
            collect(args.pmcid, args.all)
        elif args.command == 'check':
            check()
        else:
            demo()
    except (OSError, KeyError, ValueError, StopIteration) as err:
        parser.exit(1, f'Unable to complete {args.command}: {err}\n')
