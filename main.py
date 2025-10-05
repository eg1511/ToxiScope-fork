from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import json
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    XLMConfig,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

from utils import (convert_examples_to_features,
                   output_modes, processors)
from model import BERT_MLP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(train_dataset, model, tokenizer, args):
    tb_writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])

    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)

    if args['fp16']:
        scaler = torch.cuda.amp.GradScaler()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    set_seed(args['seed'])

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args['device']) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}

            with torch.cuda.amp.autocast(enabled=args['fp16']):
                outputs = model(**inputs)
                loss = outputs[0]

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                if args['fp16']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
                    optimizer.step()

                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    if args['evaluate_during_training']:
                        results, _ = evaluate(model, tokenizer, args)
                        for key, value in results.items():
                            tb_writer.add_scalar(f'eval_{key}', value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step

def evaluate(model, tokenizer, args, prefix="", test=False):
    eval_output_dir = args['output_dir']
    results = {}
    EVAL_TASK = args['task_name']
    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, args, evaluate=not test, test=test)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args['device']) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args['output_mode'] == "classification":
        probs = torch.nn.functional.softmax(torch.from_numpy(preds), dim=-1).numpy()
        pred_labels = np.argmax(preds, axis=1)
    elif args['output_mode'] == "regression":
        pred_labels = np.squeeze(preds)

    result, wrong = compute_metrics(EVAL_TASK, pred_labels, out_label_ids, probs, args)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt" if not test else "test_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results, wrong

def get_mismatched(labels, preds, examples):
    mismatched = labels != preds
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    return wrong

def get_eval_report(labels, preds, probs, args):
    f1 = f1_score(labels, preds, average='weighted')
    auc = roc_auc_score(labels, probs[:, 1], average='weighted')
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    processor = processors[args['task_name']]()
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = get_mismatched(labels, preds, examples)

    parent = [ex.text_b for ex in examples]
    text = [ex.text_a for ex in examples]

    df = pd.DataFrame(data={'text': text, 'parent': parent, 'probs': probs[:, 1], 'label': labels, 'pred': preds})
    df.to_csv(os.path.join(args['output_dir'], "test.tsv"), sep='\t', index=False)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        "roc_auc": auc
    }, wrong

def compute_metrics(task_name, preds, labels, probs, args):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds, probs, args)

def load_and_cache_examples(task, tokenizer, args, evaluate=False, test=False):
    processor = processors[task]()
    output_mode = args['output_mode']
    mode = 'test' if test else 'val' if evaluate else 'train'
    cached_features_file = os.path.join(args['data_dir'],
                                        f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")

    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_test_examples(args['data_dir']) if test else \
                   processor.get_dev_examples(args['data_dir']) if evaluate else \
                   processor.get_train_examples(args['data_dir'])

        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args['max_seq_length'],
            task=task,
            label_list=label_list,
            output_mode=output_mode
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def main():
    with open('args.json', 'r') as f:
        args = json.load(f)

    args['device'] = torch.device("cuda" if torch.cuda.is_available() and not args.get('no_cuda', False) else "cpu")
    args['n_gpu'] = torch.cuda.device_count()
    set_seed(args['seed'])

    if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
        raise ValueError(f"Output directory ({args['output_dir']}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    MODEL_CLASSES = {
        'bert': (BertConfig, BERT_MLP, BertTokenizer),
        'xlnet': (XLNetConfig, None, XLNetTokenizer), # BERT_MLP is specific to BERT
        'xlm': (XLMConfig, None, XLMTokenizer),
        'roberta': (RobertaConfig, None, RobertaTokenizer)
    }

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
    config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])
    tokenizer = tokenizer_class.from_pretrained(args['model_name'])
    model = model_class(config=config)
    model.to(args['device'])

    task = args['task_name']
    if task not in processors:
        raise ValueError(f"Task not found: {task}")

    if args['do_train']:
        train_dataset = load_and_cache_examples(task, tokenizer, args)
        global_step, tr_loss = train(train_dataset, model, tokenizer, args)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
        logger.info("Saving model checkpoint to %s", args['output_dir'])
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args['output_dir'])
        tokenizer.save_pretrained(args['output_dir'])
        torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))

    if args['do_eval']:
        checkpoints = [args['output_dir']]
        if args['eval_all_checkpoints']:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args['device'])
            evaluate(model, tokenizer, args, prefix=global_step, test=True)

if __name__ == "__main__":
    main()