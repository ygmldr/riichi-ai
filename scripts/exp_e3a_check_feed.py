#!/usr/bin/env python3
"""解析 E3a 12k 评估的 MORTAL_EVT_DEBUG JSONL,证明"分支在本臂评估中确实被投喂"。

每条 debug 记录 = 一次 Rust `evaluate()` 调用,`entries[]` 与 `react_batch` 的实参
一一对应。断言: ① 记录非空、每个 entry 的 token 行数 n >= 1;② 首 token 类型恒为
`start_kyoku`(2);③ 末 token 为合法事件类型(1..=16)。产物写入 `--out`。

用法:
  python3 scripts/exp_e3a_check_feed.py --debug /tmp/opencode/t19_e3a_evt_debug.jsonl \
      --out runs/exp/E3a/EVAL_FEED_E3A.json
"""

import argparse
import collections
import json
import os

REPO = '/home/ygsldr/riichi-ai'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--debug', required=True)
    ap.add_argument('--out', default=os.path.join(REPO, 'runs/exp/E3a/EVAL_FEED_E3A.json'))
    args = ap.parse_args()

    calls = 0
    entries = 0
    n_min = None
    n_max = 0
    first_types = collections.Counter()
    last_types = collections.Counter()
    bad = []
    with open(args.debug) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            calls += 1
            for e in rec['entries']:
                entries += 1
                n = e['n']
                if n < 1 or e['first'] != 2 or not (1 <= e['last'] <= 16):
                    bad.append({'call': rec.get('call'), **e})
                n_min = n if n_min is None else min(n_min, n)
                n_max = max(n_max, n)
                first_types[e['first']] += 1
                last_types[e['last']] += 1
    payload = {
        'debug_file': args.debug,
        'debug_size_bytes': os.path.getsize(args.debug),
        'evaluate_calls': calls,
        'entries': entries,
        'n_min': n_min,
        'n_max': n_max,
        'assertion_1_nonempty': entries > 0 and not bad,
        'assertion_2_first_is_start_kyoku': set(first_types) == {2},
        'assertion_3_last_is_valid_event_type': set(last_types) <= set(range(1, 17)),
        'failure_count': len(bad),
        'failures': bad[:10],
        'first_type_counts': dict(first_types),
        'last_type_counts': {str(k): v for k, v in sorted(last_types.items())},
        'note': (
            'entries == states queued for react_batch in the arena; every entry has n>=1, '
            'and the branch-enabled Brain raises on events=None, so all engine-evaluated '
            'decisions were fed non-empty event streams.'
        ),
    }
    assert payload['assertion_1_nonempty'], payload
    assert payload['assertion_2_first_is_start_kyoku'], payload
    assert payload['assertion_3_last_is_valid_event_type'], payload
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
