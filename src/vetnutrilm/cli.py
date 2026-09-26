"""Command-line interface: ``vetnutrilm <command> --help`` for details."""
import argparse
import hashlib
import json
import sys
from importlib import resources
from pathlib import Path

import pandas as pd

from . import __version__, data, folds, grouping, metrics, reproduce
from .parsing import parse_binary, parse_categories
from .prompts import BINARY_PROMPT_SHA256, CATEGORIES, CATEGORY_PROMPT_SHA256, TRAINING


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _new_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        sys.exit(f'Output directory {path} is not empty; use a new directory for every run')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _groups(frame):
    mapping = grouping.connected_groups(frame.id, frame.patient, frame.text)
    return frame.id.map(mapping).to_numpy()


def cmd_demo(args):
    """Run the non-GPU pipeline on invented notes. Output is not an accuracy estimate."""
    source = resources.files('vetnutrilm').joinpath('synthetic_notes.csv')
    with resources.as_file(source) as path:
        frame = data.read_records(path, 'binary')
    out = _new_dir(args.output)
    frame['group'] = _groups(frame)
    frame['fold'] = folds.binary_folds(frame.label, frame.group, k=3)
    frame[['id', 'group', 'fold']].to_csv(out / 'folds.csv', index=False)
    examples = ['[1]', '[0]', '```json\n[1]\n```', 'Yes', '[0,1,0,0,0,1]', '[0,1]']
    parsed = [dict(raw=e, binary=parse_binary(e), categories=parse_categories(e)) for e in examples]
    invented = frame.label.to_numpy().copy()
    invented[0] = 1 - invented[0]  # one deliberate error, to show the measures move
    summary = dict(notes=len(frame), linked_groups=int(frame.group.nunique()), parsing_examples=parsed,
                   illustrative_scores_on_invented_predictions=metrics.binary_scores(frame.label, invented),
                   note='Invented notes and invented predictions. Not an accuracy estimate.')
    _save(out / 'demo.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'parsing_examples'}, indent=2))
    print(f'Wrote {out}')


def cmd_prepare(args):
    """Link records by patient or identical text, then allocate grouped folds."""
    frame = data.read_records(args.input, args.task)
    frame['group'] = _groups(frame)
    if args.task == 'binary':
        frame['fold'] = folds.binary_folds(frame.label, frame.group, k=args.folds, seed=args.seed)
    else:
        sources = frame[args.source_column] if args.source_column else None
        frame['fold'] = folds.category_folds(frame[CATEGORIES].to_numpy(), frame.group, sources, k=args.folds, seed=args.seed)
    frame[['id', 'group', 'fold']].to_csv(args.output, index=False)
    print(f'{len(frame)} records, {frame.group.nunique()} linked groups, {args.folds} folds -> {args.output}')


def _encode_all(tokenizer, task, frame, with_labels):
    from . import model
    answers = data.labels(frame, task).tolist() if with_labels else [None] * len(frame)
    return [dict(id=i, **model.encode(tokenizer, task, text, labels))
            for i, text, labels in zip(frame.id, frame.text, answers)]


def cmd_train(args):
    """Train one adapter: on all records but one fold, or on every record for release."""
    from . import model
    frame = data.read_records(args.input, args.task)
    out = _new_dir(args.output)
    if args.fold:
        allocation = pd.read_csv(args.folds, dtype={'id': str}).set_index('id')
        frame['fold'] = frame.id.map(allocation.fold)
        if frame.fold.isna().any():
            sys.exit('Every record needs a fold')
        train, held = frame[frame.fold != args.fold], frame[frame.fold == args.fold]
        if set(allocation.loc[train.id, 'group']) & set(allocation.loc[held.id, 'group']):
            sys.exit('Training and held-out records share a linked group')
    else:
        train, held = frame, frame.iloc[0:0]
    base, tokenizer = model.load_base(args.checkpoint)
    rows = _encode_all(tokenizer, args.task, train, True)
    held_rows = _encode_all(tokenizer, args.task, held, False)
    longest = model.check_lengths(rows + held_rows)
    tuned, record = model.train_adapter(base, rows, out / 'adapter',
                                        progress=lambda s: print({k: round(v, 4) for k, v in s.items()}, flush=True))
    record.update(task=args.task, version=__version__, fold=args.fold, training_records=len(rows),
                  longest_tokens=longest, recipe=TRAINING,
                  prompt_sha256=BINARY_PROMPT_SHA256 if args.task == 'binary' else CATEGORY_PROMPT_SHA256,
                  adapter_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (out / 'adapter').iterdir() if p.is_file()})
    _save(out / 'training.json', record)
    if len(held_rows):
        results, speed = model.predict(tuned, tokenizer, args.task, held_rows)
        _write_predictions(results, args.task, out / 'predictions.csv')
        _save(out / 'throughput.json', speed)
    print(f'Adapter and training record written to {out}')


def cmd_classify(args):
    """Classify notes with a published or locally trained adapter."""
    from . import model
    frame = data.read_records(args.input)
    out = _new_dir(args.output)
    tasks = ['binary', 'categories'] if args.task == 'both' else [args.task]
    adapters = {'binary': args.binary_adapter, 'categories': args.category_adapter}
    merged = frame[['id']].copy()
    for task in tasks:
        if not adapters[task]:
            sys.exit(f'--{"binary" if task == "binary" else "category"}-adapter is required for {task}')
        base, tokenizer = model.load_base(args.checkpoint)
        rows = _encode_all(tokenizer, task, frame, False)
        model.check_lengths(rows)
        tuned = model.load_adapter(base, adapters[task])
        results, speed = model.predict(tuned, tokenizer, task, rows,
                                       progress=lambda n, t: print(f'{task}: {n}/{t}', flush=True))
        part = _write_predictions(results, task, out / f'{task}.csv')
        merged = merged.merge(part.drop(columns=['raw']), on='id')
        _save(out / f'{task}_throughput.json', speed)
        del tuned, base
    merged.to_csv(out / 'predictions.csv', index=False)
    print(f'Predictions written to {out / "predictions.csv"}')


def _write_predictions(results, task, path):
    rows = []
    for r in results:
        row = {'id': r['id'], 'raw': r['raw'], f'{task}_invalid': int(r['invalid'])}
        if task == 'binary':
            row['binary'] = r['prediction'][0]
        else:
            row.update({c: v for c, v in zip(CATEGORIES, r['prediction'])})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def cmd_evaluate(args):
    """Score predictions against labels with grouped bootstrap intervals."""
    frame = data.read_records(args.records, args.task)
    preds = pd.read_csv(args.predictions, dtype={'id': str}).set_index('id')
    if set(preds.index) != set(frame.id):
        sys.exit('Predictions must cover exactly the labelled records')
    groups = _groups(frame)
    invalid_column = f'{args.task}_invalid'
    invalid = int(preds[invalid_column].sum()) if invalid_column in preds else None
    if args.task == 'binary':
        y, p = frame.label.to_numpy(), preds.loc[frame.id, 'binary'].to_numpy()
        if (p < 0).any():
            sys.exit('Invalid binary outputs present; report them separately before scoring')
        point = metrics.binary_scores(y, p)
        ci = {k: metrics.grouped_interval(lambda a, b, k=k: metrics.binary_scores(a, b)[k], groups, y, p) for k in point}
    else:
        y, p = frame[CATEGORIES].to_numpy(), preds.loc[frame.id, CATEGORIES].to_numpy()
        point = metrics.category_scores(y, p)
        ci = {k: metrics.grouped_interval(lambda a, b, k=k: metrics.category_scores(a, b)[k], groups, y, p)
              for k in ('macro_f1', 'micro_f1')}
    result = dict(task=args.task, records=len(frame), linked_groups=len(set(groups.tolist())),
                  invalid_outputs=invalid, point=point, intervals_95=ci,
                  method='Pooled predictions; grouped percentile bootstrap, 2,000 replicates, seed 42')
    _save(args.output, result)
    print(json.dumps({'point': point, 'intervals_95': ci}, indent=2))


def cmd_reproduce(args):
    """Recompute the published figures from authorised run outputs."""
    binary = reproduce.binary_figures(args.binary_dataset, args.binary_results) if args.binary_dataset else None
    category = reproduce.category_figures(args.category_dataset, args.category_results) if args.category_dataset else None
    rows = reproduce.compare(binary, category)
    width = max(len(r[0]) for r in rows)
    for name, want, got, ok in rows:
        print(f'{name:<{width}}  published {want:.4f}  recomputed {got:.4f}  {"OK" if ok else "DIFFERS"}')
    failed = [r for r in rows if not r[3]]
    print(f'{len(rows) - len(failed)} of {len(rows)} figures reproduced')
    sys.exit(1 if failed else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='vetnutrilm', description='VetNutriLM: nutrition information in veterinary clinical notes')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('demo', help=cmd_demo.__doc__)
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser('prepare', help=cmd_prepare.__doc__)
    p.add_argument('--task', choices=['binary', 'categories'], required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--source-column')
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser('train', help=cmd_train.__doc__)
    p.add_argument('--task', choices=['binary', 'categories'], required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--checkpoint', required=True, help='Local Gemma-4-31B-it directory')
    p.add_argument('--output', required=True)
    p.add_argument('--folds', help='CSV from "prepare"; required with --fold')
    p.add_argument('--fold', type=int, help='Hold out this fold; omit to train on every record')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('classify', help=cmd_classify.__doc__)
    p.add_argument('--task', choices=['binary', 'categories', 'both'], required=True)
    p.add_argument('--input', required=True, help='CSV with id and text columns')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--binary-adapter')
    p.add_argument('--category-adapter')
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser('evaluate', help=cmd_evaluate.__doc__)
    p.add_argument('--task', choices=['binary', 'categories'], required=True)
    p.add_argument('--records', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser('reproduce', help=cmd_reproduce.__doc__)
    p.add_argument('--binary-dataset')
    p.add_argument('--binary-results')
    p.add_argument('--category-dataset')
    p.add_argument('--category-results')
    p.set_defaults(func=cmd_reproduce)

    args = parser.parse_args(argv)
    if args.command == 'train' and args.fold and not args.folds:
        parser.error('--fold needs --folds')
    args.func(args)


if __name__ == '__main__':
    main()
