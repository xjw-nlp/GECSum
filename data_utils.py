from torch.utils.data import Dataset
import os
import json
import random

from tqdm import tqdm
import torch
from transformers import BartTokenizer, PegasusTokenizer
import datasets
from datasets import DatasetDict
from datasets import load_from_disk


def to_cuda(batch, gpuid):
    for n in batch:
        if 'ids' in n:
            batch[n] = batch[n].to(gpuid)


class GECSumDataset(Dataset):
    def __init__(self, data_type='train', arrow_obj=None, obj_args=None):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        self.obj_args = obj_args
        self.is_pegasus = self.obj_args.is_pegasus
        self.data_type = data_type
        if data_type == 'train':
            if arrow_obj:
                if isinstance(arrow_obj, str):
                    self.arrow_data = load_from_disk(arrow_obj)[data_type]
                else:
                    self.arrow_data = arrow_obj[data_type]
            else:
                self.arrow_data = load_from_disk(self.obj_args.src_data_path)[data_type]
        else:
            self.arrow_data = load_from_disk(self.obj_args.src_data_path)[data_type]
        if self.is_pegasus:
            self.tok = PegasusTokenizer.from_pretrained(self.obj_args.model_type, verbose=False)
        else:
            self.tok = BartTokenizer.from_pretrained(self.obj_args.model_type, verbose=False)
        self.num = len(self.arrow_data)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        dp = self.arrow_data[idx]
        if self.data_type == 'train':
            src_text = dp['src_text']
            ref_text = dp['ref_text']
            cand_texts = [ref_text] + dp['cand_texts']
            cand_scores = dp['cand_scores']
            src = self.tok.batch_encode_plus([src_text], max_length=self.obj_args.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
            src_input_ids = src["input_ids"]
            src_input_ids = src_input_ids.squeeze(0)
            cand = self.tok.batch_encode_plus(cand_texts, max_length=self.obj_args.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)
            candidate_ids = cand["input_ids"]
            if self.is_pegasus:
                # add start token
                _candidate_ids = candidate_ids.new_zeros(candidate_ids.size(0), candidate_ids.size(1) + 1)
                _candidate_ids[:, 1:] = candidate_ids.clone()
                _candidate_ids[:, 0] = self.tok.pad_token_id
                candidate_ids = _candidate_ids
            result = {
                "src_input_ids": src_input_ids, 
                "candidate_ids": candidate_ids,
                "cand_scores": cand_scores,
                }
        else:
            src_text = dp[self.obj_args.feat_names[0]]
            ref_text = dp[self.obj_args.feat_names[1]]
            src = self.tok.batch_encode_plus([src_text], max_length=self.obj_args.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
            src_input_ids = src["input_ids"]
            src_input_ids = src_input_ids.squeeze(0)
            result = {
                "src_input_ids": src_input_ids, 
                "src_text": src_text, 
                "ref_text": ref_text,
                }
        return result


def collate_mp_train_gecsum(batch, pad_token_id):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)
    cand_scores = [x["cand_scores"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
        "cand_scores": cand_scores,
        }
    return result


def collate_mp_test_gecsum(batch, pad_token_id):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    src_text = [x["src_text"] for x in batch]
    ref_text = [x["ref_text"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "src_text": src_text,
        "ref_text": ref_text,
        }
    return result


def build_candidate(args, model, tokenizer, raw_data, epoch_idx, gpuid, data_type='train', bsz=4, num_samples=3200):
    if len(args.gpuid) > 1:
        _model = model.module
    else:
        _model = model
    _model.eval()
    device = f"cuda:{gpuid}"
    max_length = 140
    min_length = 55
    count = 0
    src_raw_data = raw_data[data_type]
    num_samples = min(num_samples, len(src_raw_data))
    sample_ids = random.sample(range(len(src_raw_data)), num_samples)
    res_src_texts = []
    res_ref_texts = []
    res_cand_texts = []
    res_cand_scores = []
    src_texts = []
    tgt_texts = []
    for idx in tqdm(sample_ids):
        data_point = src_raw_data[idx]
        count += 1
        src_texts.append(data_point[args.feat_names[0]])
        tgt_texts.append(data_point[args.feat_names[1]])
        if count % 100 == 0:
            print(count, flush=True)
        if count % bsz == 0:
            with torch.no_grad():
                _model.generation_mode()
                if args.config == 'xsum':
                    batch = tokenizer.prepare_seq2seq_batch(src_texts=src_texts, rmax_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True).to(device)
                    gen = _model.generate(**batch, num_return_sequences=16, num_beam_groups=16, diversity_penalty=0.1, num_beams=16, length_penalty=0.6)
                    dec  = tokenizer.batch_decode(gen, skip_special_tokens=True)
                else:
                    dct = tokenizer.batch_encode_plus(src_texts, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    summaries = _model.generate(
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
                        num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
                        max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=min_length + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        early_stopping=True,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                # print(len(dec))
                # print(dec)
            for i, hypothesis in enumerate(dec):
                hypothesis = hypothesis.replace("\n", " ")
                dec[i] = hypothesis
            _model.scoring_mode()
            encoder_input_ids = tokenizer.batch_encode_plus(dec, max_length=args.gen_max_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)["input_ids"].to(device)
            decoder_input_ids = tokenizer.batch_encode_plus(tgt_texts, max_length=args.gen_max_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)["input_ids"].to(device)
            # num_seqs = len(dec) // bsz
            # tgt_seq_len = decoder_input_ids.shape[-1]
            # decoder_input_ids = torch.repeat_interleave(decoder_input_ids, num_seqs, dim=0).contiguous().view(bsz, -1, tgt_seq_len)
            output = _model(encoder_input_ids, decoder_input_ids, args.normalize, args.score_mode, args.length_penalty, adding=args.adding, reverse=True)
            cur_pos = 0
            for batch_idx, scores in enumerate(output['score']):
                num_cands = len(scores)
                cand_set = dec[cur_pos: cur_pos + num_cands]
                paired_cand_score_arr = [(cand, score.item()) for cand, score in zip(cand_set, scores)]
                paired_cand_score_arr.sort(key=lambda x: x[1], reverse=True)
                ordered_cands, ordered_scores = map(list, zip(*paired_cand_score_arr))
                res_src_texts.append(src_texts[batch_idx])
                res_ref_texts.append(tgt_texts[batch_idx])
                res_cand_texts.append(ordered_cands)
                res_cand_scores.append(ordered_scores)
                cur_pos += num_cands
            src_texts = []
            tgt_texts = []
    if src_texts:
        with torch.no_grad():
            _model.generation_mode()
            if args.config == 'xsum':
                batch = tokenizer.prepare_seq2seq_batch(src_texts=src_texts, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True).to(device)
                gen = _model.generate(**batch, num_return_sequences=16, num_beam_groups=16, diversity_penalty=0.1, num_beams=16, length_penalty=0.6)
                dec  = tokenizer.batch_decode(gen, skip_special_tokens=True)
            else:
                dct = tokenizer.batch_encode_plus(src_texts, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                summaries = _model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
                    max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True,
                )
                dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            # print(len(dec))
            # print(dec)
        for i, hypothesis in enumerate(dec):
            hypothesis = hypothesis.replace("\n", " ")
            dec[i] = hypothesis
        _model.scoring_mode()
        encoder_input_ids = tokenizer.batch_encode_plus(dec, max_length=args.gen_max_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)["input_ids"].to(device)
        decoder_input_ids = tokenizer.batch_encode_plus(tgt_texts, max_length=args.gen_max_len, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)["input_ids"].to(device)
        # num_seqs = len(dec) // bsz
        # tgt_seq_len = decoder_input_ids.shape[-1]
        # decoder_input_ids = torch.repeat_interleave(decoder_input_ids, num_seqs, dim=0).contiguous().view(bsz, -1, tgt_seq_len)
        output = _model(encoder_input_ids, decoder_input_ids, args.normalize, args.score_mode, args.length_penalty, adding=args.adding, reverse=True)
        cur_pos = 0
        for batch_idx, scores in enumerate(output['score']):
            num_cands = len(scores)
            cand_set = dec[cur_pos: cur_pos + num_cands]
            paired_cand_score_arr = [(cand, score.item()) for cand, score in zip(cand_set, scores)]
            paired_cand_score_arr.sort(key=lambda x: x[1], reverse=True)
            ordered_cands, ordered_scores = map(list, zip(*paired_cand_score_arr))
            res_src_texts.append(src_texts[batch_idx])
            res_ref_texts.append(tgt_texts[batch_idx])
            res_cand_texts.append(ordered_cands)
            res_cand_scores.append(ordered_scores)
            cur_pos += num_cands
    _model.train()
    dataset = datasets.Dataset.from_dict({'src_text': res_src_texts, 'ref_text': res_ref_texts, 'cand_texts': res_cand_texts, 'cand_scores': res_cand_scores})
    arrow_obj = DatasetDict({data_type: dataset})
    arrow_path = os.path.join(args.dataset, f"{args.config}_epoch_{epoch_idx}_gpu_{gpuid}")
    arrow_obj.save_to_disk(arrow_path)
    return arrow_obj
