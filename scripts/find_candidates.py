import pandas as pd, ast, sys

CURRENT = {
    'a5a51ad3a1200e2e5ef46c140bab717422e41ca2',
    'b12aa1d44352de21d1a6faaf04172d8c2508b42b',
    'bc6e0c471c4d7d6cd150149f2830e9d23a0040bc',
    '3b9c747d71f30c6a59f6529f8475d7f56a86a7c5',
    '19dd9342e7bc55c877367b7474caf41e819e38c3',
    'e961e063a27863b0505b59219e38f83450a6831d',
    '7b8db6083b34520688dbc71f341f7aeaf156bf17',
    'e7f497570abb6b4ae5af4970620cd880e4c0c904',
    '4213ac97be449d0e40631a314d2b7bd3901d4967',
    '611d80db29dd7b0cfb755772c69d60ae5bca05f9',
    '8722aeebdf823763596869a71eb6a7077bff7ccf',
    'ec81825aaf7e848d9f8ddffdf1e0d20aebe9172c',
    'fca9874a9b42a2134f907d2fb46ab774a831404a',
}

vcc = pd.read_csv('data/processed/cve_fc_vcc_mapping.csv')
fi  = pd.read_csv('data_new/processed/file_info_new.csv', usecols=['hash','filename'], low_memory=False)
ci  = pd.read_csv('data/processed/commit_info.csv', usecols=['hash','commit_type','author','author_date'])

known = set(fi['hash'].dropna().str.strip())
tf = vcc[vcc['repo_url'].str.contains('tensorflow', na=False)].copy()

seen = set()
results = []
for _, row in tf.iterrows():
    cve = row['cve_id']
    fc  = str(row['fc_hash']).strip()
    try:
        vccs = list(dict.fromkeys(ast.literal_eval(row['vcc_hash'])))
    except:
        continue
    if not vccs:
        continue
    key = (cve, fc)
    if key in seen:
        continue
    seen.add(key)
    all_h = set([fc] + vccs)
    all_in = all_h <= known
    new = all_h - CURRENT
    results.append(dict(cve=cve, fc=fc, vccs=vccs,
                        n=len(all_h), all_in=all_in,
                        new_n=len(new), new=sorted(new)))

full = sorted([r for r in results if r['all_in']], key=lambda r: r['new_n'])
print(f"Fully resolvable TF CVE chains: {len(full)}")
for r in full:
    print(f"\n  {r['cve']}  total={r['n']}  new={r['new_n']}")
    print(f"    FC : {r['fc']}")
    print(f"    VCC: {r['vccs']}")
    for h in r['new']:
        fnames = fi[fi['hash']==h]['filename'].str.split('/').str[-1].tolist()
        ci_row = ci[ci['hash']==h]
        ctype  = ci_row['commit_type'].values[0] if len(ci_row) else '?'
        author = ci_row['author'].values[0].split('@')[0] if len(ci_row) else '?'
        date   = ci_row['author_date'].values[0][:10] if len(ci_row) else '?'
        print(f"      {h[:16]} ({ctype}) {author} {date}: {fnames[:4]}")

# Clean (non-CVE) commits touching the same files
target_files = {'conv_ops.cc','conv_grad_ops.cc','conv_grad_shape_utils.cc','sparse_matmul_op.cc'}
fi['basename'] = fi['filename'].str.split('/').str[-1]
fi_target = fi[fi['basename'].isin(target_files)]
all_cve_h = set()
for r in results:
    all_cve_h |= set([r['fc']] + r['vccs'])
clean_cands = set(fi_target['hash'].dropna()) - CURRENT - all_cve_h
ci_clean = ci[ci['hash'].isin(clean_cands)].sort_values('author_date')
print(f"\n=== Clean (non-CVE) commits on target files: {len(ci_clean)} ===")
for _, row in ci_clean.iterrows():
    fnames = fi[fi['hash']==row['hash']]['basename'].tolist()
    print(f"  {row['hash']}  {str(row['commit_type']):5}  {row['author'].split('@')[0][:20]:20}  {row['author_date'][:10]}  {fnames[:3]}")
