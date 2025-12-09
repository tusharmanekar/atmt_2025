import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        # Forward pass: use only the generated tokens so far
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens

def length_penalty(seq_len: int, alpha: float) -> float:
    lp = ((5 + seq_len) ** alpha) / (6 ** alpha)
    return lp

def lp_scoring(seq: torch.Tensor, raw_score: float, alpha: float) -> float:
    # number of target tokens excluding BOS therefore we do -1 or it throws error
    seq_len = seq.size(1) - 1
    lp = length_penalty(seq_len, alpha)
    return raw_score / lp

def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):
    """Beam Search decoding compatible with Transformer-based Seq2Seq models."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    # __QUESTION 1: what does this line set up and why is the beam represented this way?
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]
                # __QUESTION 2: Why do we need to create trg_pad_mask here and how does it affect the model's predictions?
                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                # __QUESTION 3: Explain the purpose of applying log_softmax and selecting top-k tokens here.
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                # __QUESTION 4: explain the tensor shapes and the logic when creating new_seq and new_score below. Is any broadcasting or indexing issue possible?
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        beams = sorted(new_beams, key=lambda beam: lp_scoring(beam[0], beam[1], alpha), reverse=True)[:beam_size]

        # __QUESTION 5: Why do we check for EOS here and what does it imply for beam search?
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break
    best_seq, _ = beams[0]
    # __QUESTION 6: What is returned, and why are we squeezing, converting to list and wrapping in another list here?
    return [best_seq.squeeze(0).tolist()]

def beam_search_decode_relative_prun(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    args,
    device: torch.device,
    beam_size: int = 5,
    alpha: float = 0.7,
    rp: float = 0.6,   # relative pruning threshold (in log-prob units)
):
    """
    Beam search with length normalization and relative threshold pruning
    (Freitag et al., 2017).
    """
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]

                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_token = topk_ids[:, k].unsqueeze(0)  # (1, 1)
                new_seq = torch.cat([seq, new_token], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # Relative threshold pruning
        # Compute normalized scores for all new beams
        norm_scores = [
            lp_scoring(seq, raw_score, alpha) for (seq, raw_score) in new_beams
        ]
        max_score = max(norm_scores)

        pruned_beams = []
        for (beam, norm_s) in zip(new_beams, norm_scores):
            if norm_s >= rp * max_score:
                pruned_beams.append(beam)

        # If everything got pruned, keep at least best
        if not pruned_beams:
            best_idx = max(range(len(new_beams)), key=lambda i: norm_scores[i])
            pruned_beams = [new_beams[best_idx]]

        # Sort remaining beams by normalized score and keep up to beam_size
        beams = sorted(
            pruned_beams,
            key=lambda beam: lp_scoring(beam[0], beam[1], alpha),
            reverse=True
        )[:beam_size]

        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    best_seq, _ = beams[0]
    return [best_seq.squeeze(0).tolist()]

def beam_search_decode_absolute_prun(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    args,
    device: torch.device,
    beam_size: int = 5,
    alpha: float = 0.7,
    ap: float = 2.0,   # absolute pruning threshold (in log-prob units)
):
    """
    Beam search with length normalization and absolute threshold pruning
    (Freitag et al., 2017).
    """
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]

                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_token = topk_ids[:, k].unsqueeze(0)
                new_seq = torch.cat([seq, new_token], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # Absolute threshold pruning
        norm_scores = [
            lp_scoring(seq, raw_score, alpha) for (seq, raw_score) in new_beams
        ]
        max_score = max(norm_scores)

        pruned_beams = []
        for (beam, norm_s) in zip(new_beams, norm_scores):
            if norm_s >= max_score - ap:
                pruned_beams.append(beam)

        if not pruned_beams:
            best_idx = max(range(len(new_beams)), key=lambda i: norm_scores[i])
            pruned_beams = [new_beams[best_idx]]

        beams = sorted(
            pruned_beams,
            key=lambda beam: lp_scoring(beam[0], beam[1], alpha),
            reverse=True
        )[:beam_size]

        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    best_seq, _ = beams[0]
    return [best_seq.squeeze(0).tolist()]

