"""Read-only eligibility catalogue, stratified exactly like the original data."""
import itertools
import math
from common import *

def angle(a, b):
    return abs((float(a)-float(b)+180) % 360-180)

def compatible_choices(plan):
    src = {s['source_id']:s for s in plan['sources']}
    static = [k for k,s in src.items() if s['trajectory']['type']=='static']
    linear = [k for k,s in src.items() if s['trajectory']['type']=='linear']
    def separated(ids):
        return all(angle(src[a]['trajectory']['position']['azimuth_deg'],src[b]['trajectory']['position']['azimuth_deg']) >= 45
                   for a,b in itertools.combinations(ids,2))
    swaps = [list(ids) for ids in itertools.combinations(static,2) if separated(ids)]
    cycles = [list(ids) for ids in itertools.combinations(static,3) if separated(ids)]
    return {'static':static, 'linear':linear, 'swaps':swaps, 'cycles':cycles}

def eligible(choices, task):
    s,l = len(choices['static']),len(choices['linear'])
    return {'diagonal_relocation':s>=1, 'diagonal_start_motion':s>=1,
            'dual_relocation':s>=2, 'position_swap':bool(choices['swaps']),
            'dual_start_motion':s>=2, 'dual_stop_motion':l>=2,
            'mixed_motion_toggle':s>=1 and l>=1,
            'three_position_cycle':bool(choices['cycles'])}[task]

def build(split):
    path = ROOT/'catalog'/f'{split}.sqlite'
    marker = path.with_suffix('.json')
    if marker.exists():
        saved = read(marker)
        assert sha(path)==saved['sha256']; return saved
    original = BASE/'training_index'/f'{split}.sqlite'
    frozen = read(str(original)+'.frozen.json')
    assert sha(original)==frozen['index_sha256']
    src = db(original)
    strata = json.loads(src.execute("SELECT value FROM metadata WHERE key='source_strata_counts_json'").fetchone()[0])
    quotas = allocation(strata, COUNTS[split])
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(f'.tmp.{os.getpid()}')
    if temporary.exists(): temporary.unlink()
    out = sqlite3.connect(temporary)
    out.executescript('''PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE sources(ordinal INTEGER PRIMARY KEY,sample_id TEXT UNIQUE,stratum TEXT,rank INTEGER,
          choices_json TEXT,master_gain REAL);
        CREATE INDEX source_stratum_rank ON sources(stratum,rank);
        CREATE TABLE eligibility(task TEXT,stratum TEXT,ordinal INTEGER,rank INTEGER,PRIMARY KEY(task,ordinal)) WITHOUT ROWID;
        CREATE INDEX eligible_stratum_rank ON eligibility(task,stratum,rank);''')
    pending=[]; ep=[]; n=0
    for r in src.execute('SELECT pair_ordinal,source_sample_id,source_count,source_domain,latent_bucket_frames,old_sceneplan_zlib,source_render_result_zlib FROM pairs ORDER BY pair_ordinal'):
        old = unpack(r['old_sceneplan_zlib']); choice = compatible_choices(old)
        key = f"{r['source_count']}|{r['source_domain']}|{r['latent_bucket_frames']}"
        rank = seed(split,r['source_sample_id'],'catalog') % (2**63-1)
        master = unpack(r['source_render_result_zlib'])['master_gain']
        assert math.isfinite(master) and master > 0
        pending.append((r['pair_ordinal'],r['source_sample_id'],key,rank,canonical(choice),master))
        ep.extend((task,key,r['pair_ordinal'],rank) for task in TRAIN_TASKS if eligible(choice,task))
        n += 1
        if n%2000==0:
            out.executemany('INSERT INTO sources VALUES(?,?,?,?,?,?)',pending)
            out.executemany('INSERT INTO eligibility VALUES(?,?,?,?)',ep)
            out.commit();pending.clear();ep.clear()
        if n%100000==0: state('catalog',split=split,scanned=n,total=frozen['rows'])
    if pending:
        out.executemany('INSERT INTO sources VALUES(?,?,?,?,?,?)',pending)
        out.executemany('INSERT INTO eligibility VALUES(?,?,?,?)',ep);out.commit()
    assert n==frozen['rows']
    counts={f'{task}|{key}':count for task,key,count in out.execute('SELECT task,stratum,count(*) FROM eligibility GROUP BY task,stratum')}
    assert out.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    out.close();src.close();temporary.replace(path)
    result=dict(status='PASS_ORIGINAL_ELIGIBILITY_CATALOGUE',split=split,rows=n,source_index=str(original),
                source_index_sha256=frozen['index_sha256'],joint_quotas=quotas,eligibility=counts,
                maximum_new_pairs_per_original_source=2,sha256=sha(path),created_at=now())
    write(marker,result); return result

if __name__=='__main__':
    for split in COUNTS:build(split)
