"""CAD-TopoBench dataset record schema and IO.

One benchmark record per sample (JSONL):

{
  "id":        "00000093",                      # sample id (DeepCAD test split)
  "source":    "deepcad_test",
  "gt":  {"success": true, "betti": [1,4,0], "h1": 4},
  "gen": {"success": true, "betti": [3,6,0], "h1": 6},
  "candidates": [                               # top-k sampling track (F3)
      {"cand_idx": 0, "temperature": 0.0001, "h1": 6, "success": true}, ...
  ],
  "geometry": {                                 # filled by geometry stage (F1/T2)
      "computed": false, "cd": null, "f_score_0.01": null
  },
  "failures": ["hallucinated_holes", ...]       # filled by metrics stage (F2/T1)
}
"""
import json
import os

from .paths import EXTERNAL

PH_DIR = str(EXTERNAL / 'ph_results')
RESULTS_DIR = str(EXTERNAL / 'DeepCAD' / 'proj_log' / 'pretrained' / 'pretrained' / 'results')

RECORD_FIELDS = ['id', 'source', 'gt', 'gen', 'candidates', 'geometry', 'failures']


def normalize_id(data_id):
    """'0025/00250456' -> '00250456'."""
    return data_id.split('/')[-1]


def _load_json(name):
    with open(os.path.join(PH_DIR, name), 'r') as f:
        return json.load(f)


def build_records(ph_dir=PH_DIR):
    """Merge existing DeepCAD experiment outputs into unified benchmark records."""
    global PH_DIR
    PH_DIR = ph_dir

    gt_results = _load_json('test_topology.json')
    gen_results = _load_json('generated_topology.json')
    topk_results = _load_json('topk_topology.json')

    gt_dict = {normalize_id(r['data_id']): r for r in gt_results}
    gen_dict = {normalize_id(r['data_id']): r for r in gen_results}
    topk_dict = {r['sample_id']: r for r in topk_results}

    all_ids = sorted(set(gt_dict) | set(gen_dict) | set(topk_dict))
    records = []
    for sid in all_ids:
        g = gt_dict.get(sid)
        p = gen_dict.get(sid)
        tk = topk_dict.get(sid)
        rec = {
            'id': sid,
            'source': 'deepcad_test',
            'gt': {
                'success': bool(g and g.get('success')),
                'betti': g.get('betti') if g else None,
                'h1': g.get('h1') if g else None,
            },
            'gen': {
                'success': bool(p and p.get('success')),
                'betti': p.get('betti') if p else None,
                'h1': p.get('h1') if p else None,
            },
            'candidates': tk.get('candidates') if tk else [],
            'geometry': {'computed': False, 'cd': None, 'f_score_0.01': None},
            'failures': [],
        }
        records.append(rec)
    return records


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')


def load_jsonl(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
