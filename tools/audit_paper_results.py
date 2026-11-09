#!/usr/bin/env python3
"""Validate canonical paper inputs and emit paired free-form uncertainty tables.

Read-only with respect to measurements. Bootstrap resamples the same question indices
across models, separately within each of the seven fixed ten-item buckets. Intervals
condition on the recorded ten-grade means; they do not estimate judge/vendor bias.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict

import numpy as np

from paper_tables import FAIR, ROOT, SCORED_UNANSWERED_PREFIX, load, _is_correct
from stats_tests import mcnemar_exact

METRICS = ('fact_correct', 'hallucination_free', 'source_grounded', 'composite')


def main() -> None:
    """Fail on invalid coverage/scoring, then regenerate audit and LaTeX tables."""
    output = ROOT / 'paper/article/generated'
    output.mkdir(parents=True, exist_ok=True)
    report = []
    rows = []
    gold = {q['id']: q['answer'] for p in sorted((ROOT / 'data/mcq').glob('mcp*.json'))
            for q in json.loads(p.read_text())}
    assert len(gold) == 400
    for tier, expected_cells in [('small', 84), ('medium', 55)]:
        path = ROOT / f'results/{tier}/data/raw_results.csv'
        rs = load(path)
        cells = defaultdict(list)
        for r in rs:
            cells[r['model_label'], r['retrieval_condition']].append(r)
            assert r['question_id'] in gold
            assert r['predicted_letter'] in 'ABCDEFX' and len(r['predicted_letter']) == 1
            assert _is_correct(r) == (r['predicted_letter'] == gold[r['question_id']])
        assert len(cells) == expected_cells
        for key, cell in cells.items():
            assert len(cell) == 400 and {r['question_id'] for r in cell} == set(gold), key
        report.append(f'{tier}: {len(rs)} rows; {len(cells)} complete cells; keys and correctness verified; SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}')
        rows.extend(r for r in rs if r['retrieval_condition'] in FAIR)
    assert len(rows) == 51200
    report.append(f"Escalated: {sum(int(r.get('attempts') or 1) > 1 for r in rows)}")
    failures = Counter(r['error'].rsplit('(', 1)[-1].rstrip(')') for r in rows
                       if r.get('error', '').startswith(SCORED_UNANSWERED_PREFIX))
    report.append(f'Unanswered: {dict(failures)}; total {sum(failures.values())}')
    ff = load(ROOT / 'results/freeform/graded.csv')
    answers = load(ROOT / 'results/freeform/answers.csv')
    answer_map = {(r['model_label'], r['question_id']): r for r in answers}
    assert len(answers) == len(answer_map) == 420
    for r in ff:
        original = answer_map[r['model_label'], r['question_id']]
        assert all(r[k] == v for k, v in original.items()), 'Grade/answer mismatch'
    report.append('Free-form: 420 unique answers match graded source fields exactly; all have ten grades.')
    questions = json.loads((ROOT / 'data/freeform/questions.json').read_text())
    if isinstance(questions, dict):
        questions = questions['questions']
    qids = sorted(q['id'] for q in questions)
    assert len(ff) == 420 and len(qids) == 70
    by = defaultdict(dict)
    for r in ff:
        assert r['question_id'] not in by[r['model_label']]
        assert int(r['n_grades']) == 10
        assert all(0 <= float(r[m]) <= 1 for m in METRICS[:3])
        by[r['model_label']][r['question_id']] = r
    assert len(by) == 6
    for model, rs in by.items():
        assert set(rs) == set(qids), model
        assert sorted(Counter(r['bucket'] for r in rs.values()).values()) == [10]*7
    models = sorted(by)
    values = np.array([[[float(by[m][q][metric]) for metric in METRICS[:3]]
                        for q in qids] for m in models])
    values = np.concatenate([values, values.mean(axis=2, keepdims=True)], axis=2)
    rng = np.random.default_rng(42)
    bucket_ids = defaultdict(list)
    for i, q in enumerate(qids):
        bucket_ids[by[models[0]][q]['bucket']].append(i)
    indices = np.concatenate([rng.choice(ids, size=(20000, len(ids)), replace=True)
                              for _, ids in sorted(bucket_ids.items())], axis=1)
    samples = values[:, indices, :].mean(axis=2) * 100
    means = values.mean(axis=1) * 100
    names = {m: by[m][qids[0]]['model'] for m in models}
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & Fact-correct & Hallucination-free & Source-grounded & Composite \\', r'\midrule']
    for i in sorted(range(6), key=lambda i: -means[i, 3]):
        lines.append(names[models[i]] + ' & ' + ' & '.join(f'{v:.1f}\\%' for v in means[i]) + r' \\')
        report.append(f'{names[models[i]]}: means={means[i].round(4).tolist()}, k={sorted({r["top_k"] for r in by[models[i]].values()})}')
    lines.extend([r'\midrule', 'Overall & ' + ' & '.join(f'{v:.1f}\\%' for v in means.mean(axis=0)) + r' \\', r'\bottomrule', r'\end{tabular}'])
    (output/'freeform.tex').write_text('\n'.join(lines)+'\n')
    base = models.index('gemma4_e4b_thinking_on')
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Model & Composite & 95\% CI & Difference from E4B [95\% CI] \\', r'\midrule']
    for i in sorted(range(6), key=lambda i: -means[i, 3]):
        lo, hi = np.percentile(samples[i, :, 3], [2.5, 97.5])
        dlo, dhi = np.percentile(samples[i, :, 3] - samples[base, :, 3], [2.5, 97.5])
        diff = means[i, 3] - means[base, 3]
        lines.append(f'{names[models[i]]} & {means[i,3]:.1f} & [{lo:.1f}, {hi:.1f}] & {diff:+.1f} [{dlo:+.1f}, {dhi:+.1f}]' + r' \\')
        report.append(f'{names[models[i]]}: composite CI [{lo:.4f}, {hi:.4f}]; paired difference {diff:.4f} [{dlo:.4f}, {dhi:.4f}]')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (output/'freeform_uncertainty.tex').write_text('\n'.join(lines)+'\n')
    for i,j in itertools.combinations(range(6), 2):
        for k, metric in enumerate(METRICS):
            ci = np.percentile(samples[i,:,k]-samples[j,:,k], [2.5,97.5])
            report.append(f'PAIR {names[models[i]]} - {names[models[j]]}, {metric}: {means[i,k]-means[j,k]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]')
    i = models.index('gemma4_12b_thinking_on')
    for k,metric in enumerate(METRICS[:3]):
        test = mcnemar_exact(dict(zip(qids, values[base,:,k] >= .5)), dict(zip(qids, values[i,:,k] >= .5)))
        report.append(f'E4B vs 12B McNemar {metric}: {test}')
    mcq = []
    for m in models:
        mcq.append(max(np.mean([_is_correct(r) for r in rows if r['model_label']==m and r['retrieval_condition']==c])*100 for c in FAIR if c != 'no_rag'))
    def ranks(a):
        return np.array([1+sum(a<v)+.5*(sum(a==v)-1) for v in a])
    x, y = ranks(np.array(mcq)), ranks(means[:,2])
    rho = float(np.corrcoef(x,y)[0,1])
    perm = [np.corrcoef(x, p)[0,1] for p in itertools.permutations(y)]
    p = np.mean(np.abs(perm) >= abs(rho)-1e-12)
    report.append(f'Spearman MCQ vs grounding: rho={rho:.6f}, exact permutation p={p:.6f}; n=6; 720 permutations')
    (ROOT/'results/rerun_logs/paper_audit.txt').write_text('\n'.join(report)+'\n')
    print('\n'.join(report[:18] + report[-4:]))


if __name__ == '__main__':
    main()
