#!/usr/bin/env python3
"""Read-only inventory of Baltic Monte Carlo run directories across drives.

Finds run_XXXX_seed_YYYY folders, checks the SAME structural CSV completion
rule as run_may_exact.py, and independently looks for dated May 2026 data.
Never copies, deletes, repairs, or launches a simulation.
"""
import argparse
import csv
import os
from pathlib import Path
import re
import sys

RUN_RE = re.compile(r'^run_(\d+)_seed_(\d+)$', re.I)
DATE_RE = re.compile(r'(?<!\d)(20\d\d-\d\d)-\d\d(?!\d)')
STAGES = ('results_S1_DA_only','results_S2_DA_MFRR_energy',
          'results_S3_reserve_competition_full_stack')
PRUNE = {
    '$recycle.bin', 'system volume information', '$windows.~bt',
    'windows', 'program files', 'program files (x86)', 'programdata',
    '.git', '.venv', 'venv', 'env', 'node_modules', '__pycache__',
    'site-packages', '.cache', '.npm', 'anaconda3', 'miniconda3',
}


def csv_rows(path):
    if not path.is_file():
        return None
    try:
        with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as fh:
            reader = csv.reader(fh)
            next(reader, None)
            return sum(1 for row in reader if row and any(c.strip() for c in row))
    except (OSError, csv.Error):
        return -1


def detect_month(run, summary_paths):
    """Date evidence from actual output CSVs or actual per-day result folders."""
    months = set()
    for f in summary_paths:
        if f.is_file():
            try:
                with f.open('r', encoding='utf-8-sig', errors='replace') as fh:
                    sample = fh.read(120_000)
                months.update(DATE_RE.findall(sample))
            except OSError:
                pass
        if f.parent.is_dir():
            try:
                with os.scandir(f.parent) as children:
                    for child in children:
                        if child.is_dir(follow_symlinks=False):
                            months.update(DATE_RE.findall(child.name))
            except OSError:
                pass
    if not months:
        return 'UNVERIFIED', ''
    if months == {'2026-05'}:
        return 'MAY_2026', ','.join(sorted(months))
    if months == {'2025-11'}:
        return 'NOV_2025', ','.join(sorted(months))
    return 'MIXED_OR_OTHER', ','.join(sorted(months))


def examine(path):
    match = RUN_RE.fullmatch(path.name)
    folder_index, seed = map(int, match.groups())
    inferred_index = seed - 1010 + 1
    summaries = [path/'results'/stage/'monthly_chain_summary.csv' for stage in STAGES]
    counts = [csv_rows(f) for f in summaries]
    comparison = csv_rows(path/'comparison.csv')
    totals = csv_rows(path/'scenario_totals.csv')
    exact_complete = comparison == 1 and totals == 3 and counts == [31,31,31]
    month, evidence = detect_month(path, summaries)
    worker_local = '_local' in (p.lower() for p in path.parts)
    return {
        'inferred_base_from_seed': inferred_index,
        'folder_base': folder_index,
        'seed': seed,
        'file_complete_31_day_rule': 'YES' if exact_complete else 'NO',
        'month_from_output_dates': month,
        'months_detected': evidence,
        'comparison_rows': comparison,
        'scenario_totals_rows': totals,
        'S1_daily_rows': counts[0],
        'S2_daily_rows': counts[1],
        'S3_daily_rows': counts[2],
        'worker_local_copy': 'YES' if worker_local else 'NO',
        'path': str(path),
    }


def scan(roots):
    records = []
    dirs_seen = 0
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            print('[SKIP] Root not accessible:', root, flush=True)
            continue
        print('[SCAN]', str(root), flush=True)
        def on_error(_exc):
            pass
        for current, dirnames, _filenames in os.walk(root, topdown=True,
                                                     followlinks=False, onerror=on_error):
            dirs_seen += 1
            if dirs_seen % 5000 == 0:
                print(f'[PROGRESS] {dirs_seen:,} folders visited, {len(records)} run folders found', flush=True)
            here = Path(current)
            m = RUN_RE.fullmatch(here.name)
            if m:
                record = examine(here)
                records.append(record)
                # Only _local can contain another actual run folder. Daily
                # result directories can contain thousands of files: skip them.
                dirnames[:] = [d for d in dirnames if d.lower() == '_local']
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in PRUNE]
    return records, dirs_seen


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        writer=csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', nargs='+' , default=['.'],
                        help='Directories to inspect')
    parser.add_argument('--out', default='baltic_result_audit',
                        help='New report directory (NOT inside final_results)')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if any(part.lower() in {'final_results','_workspaces_final_results'} for part in out.parts):
        parser.error('Choose an output location OUTSIDE final_results and _workspaces_final_results.')
    records, dirs_seen = scan(args.root)
    out.mkdir(parents=True, exist_ok=True)
    fields = ['inferred_base_from_seed','folder_base','seed','file_complete_31_day_rule',
              'month_from_output_dates','months_detected','comparison_rows',
              'scenario_totals_rows','S1_daily_rows','S2_daily_rows','S3_daily_rows',
              'worker_local_copy','path']
    write_csv(out/'all_found_run_folders.csv', records, fields)
    report = []
    for base in range(1,114):
        all_for_seed = [r for r in records if r['seed'] == 1009+base]
        verified = [r for r in all_for_seed if r['file_complete_31_day_rule']=='YES'
                    and r['month_from_output_dates']=='MAY_2026']
        # A 31-day result whose month isn't explicit must be reviewed, not copied.
        possible = [r for r in all_for_seed if r['file_complete_31_day_rule']=='YES'
                    and r['month_from_output_dates']=='UNVERIFIED']
        incomplete_may = [r for r in all_for_seed if r['month_from_output_dates']=='MAY_2026'
                          and r['file_complete_31_day_rule']=='NO']
        if verified:
            status = 'MAY_COMPLETE_FOUND'
            shown = verified
        elif possible:
            status = 'COMPLETE_BUT_MONTH_UNVERIFIED'
            shown = possible
        elif incomplete_may:
            status = 'MAY_PARTIAL_FOUND'
            shown = incomplete_may
        elif all_for_seed:
            status = 'OTHER_MONTH_OR_INCOMPLETE_ONLY'
            shown = all_for_seed
        else:
            status = 'NO_RUN_FOLDER_FOUND'
            shown = []
        report.append({'base':base,'expected_seed':1009+base,'status':status,
                       'candidate_count':len(all_for_seed),
                       'best_candidate_paths':' | '.join(r['path'] for r in shown[:5]),
                       'all_matching_paths':' | '.join(r['path'] for r in all_for_seed[:15])})
    write_csv(out/'may_1_to_113_coverage.csv',report,
              ['base','expected_seed','status','candidate_count',
               'best_candidate_paths','all_matching_paths'])
    print(f'\n[SEARCH DONE] Folders visited: {dirs_seen:,}; run folders found: {len(records)}')
    from collections import Counter
    counts = Counter(row['status'] for row in report)
    for k,v in sorted(counts.items()):
        print(f'  {k}: {v}')
    for status in ('MAY_COMPLETE_FOUND','COMPLETE_BUT_MONTH_UNVERIFIED',
                   'MAY_PARTIAL_FOUND','OTHER_MONTH_OR_INCOMPLETE_ONLY','NO_RUN_FOLDER_FOUND'):
        bases = [x['base'] for x in report if x['status']==status]
        if bases:
            print(f'[{status}] Bases:', ', '.join(map(str,bases)))
    print('[REPORT]', out/'may_1_to_113_coverage.csv')
    print('[DETAIL]', out/'all_found_run_folders.csv')
    print('[SAFE] Read-only search. No simulation, copying, or changes to existing results.')


if __name__=='__main__':
    main()
