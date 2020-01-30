# coding: utf-8
# modified https://github.com/joeynmt/joeynmt/blob/master/joeynmt/search.py to work with trax
import numpy as np


def transformer_greedy(model, batch, max_output_length):
    """
    Greedy decoder for the transformer

    :param model: Transformer EncDec model
    :param batch: a batch of (src, tgt) objects
    :param: max_output_length: maximum length for the hypotheses
    :return:
        - ys: output hypotheses (2d array of indices),
    """

    bos_index, eos_index = 0, 1
    src, _ = batch
    batch_size = src.shape[0]
    ys = np.zeros((batch_size, 1), dtype=np.int32)
    finished = np.zeros((batch_size), dtype=np.uint8)

    for i in range(max_output_length):

        logits, _ = model((src, ys))
        logits = logits[:, -1]
        next_word = np.argmax(logits, axis=1)
        if i == 0:
            # doing this because of how shiftRight works
            ys = np.reshape(next_word, (batch_size, 1))
        else:
            ys = np.concatenate((ys, np.expand_dims(next_word, axis=1)), axis=1)

        is_eos = np.equal(next_word, eos_index)
        finished += is_eos

        if (finished >= 1).sum() == batch_size:
            break

    return ys.copy()

# pylint: disable=too-many-statements,too-many-branches
def transformer_batch(
        model, batch, size, max_output_length, alpha=-1, n_best=1):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param model:
    :param size: size of the beam
    :param pad_index:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
    """
    assert size > 0, 'Beam size must be >0.'
    assert n_best <= size, 'Can only return {} best hypotheses.'.format(size)
    import pdb;pdb.set_trace()

    # init
    src, _ = batch
    # TODO(alex-fabbri): remove hard coding
    box_index, pad_index, eos_index = 0, 0, 1
    batch_size = src.shape[0]

    # how many times you want that dimension to appear in the final ar
    src = np.tile(src, (size, 1))
    # numbering elements in the batch
    batch_offset = np.arange(batch_size, dtype=np.int64)
    # numbering elements in the extended batch, i.e. beam size copies of each
    # batch element
    beam_offset = np.arange(0, batch_size * size, step=size, dtype=np.int64)

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = np.full((batch_size * size, 1), box_index, dtype=np.int64)

    # Give full probability to the first beam on the first step.
    topk_log_probs = np.zeros((batch_size, size), dtype=np.float32)
    topk_log_probs[:, 1:] = float("-inf")
    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):

        # For Transformer, we feed the complete predicted sentence so far.
        decoder_input = alive_seq # complete prediction so far

        # expand current hypotheses
        # decode one single step
        # logits: logits for final softmax
        log_probs, _ = model((src, decoder_input))
        log_probs = log_probs[:, -1]

        # multiply proobs by the beam probability (=add logprobs)
        log_probs += np.expand_dims(np.reshape(topk_log_probs, -1), -1)
        curr_scores = log_probs

        # compute length penalty
        # TODO(alex-fabbri): check what alpha should be set to
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        # TODO(alex-fabbri): remove hard coding
        curr_scores = curr_scores.reshape(-1, size * 32000)

        # pick currently best top k hypotheses (flattened order)
        topk_ids = np.argsort(curr_scores, axis=1)[:, -size:]
        topk_scores = curr_scores[topk_ids[:, :], topk_ids[:, :]]

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores

        # reconstruct beam origin and true word ids from flattened order
        # TODO(alex-fabbri): remove hard coding
        topk_beam_index = np.floor_divide(topk_ids, 32000)
        topk_ids = np.fmod(topk_ids, 32000)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + np.expand_dims(beam_offset[:topk_beam_index.shape[0]], axis=1))
        select_indices = np.reshape(batch_index, -1)

        # append latest prediction
        alive_seq = np.concatenate(
            [np.take(alive_seq, select_indices, axis=0),
             np.reshape(topk_ids, (-1, 1))], -1)  # batch_size*k x hyp_len

        # is_finished = topk_ids.equal(eos_index)
        is_finished = np.equal(topk_ids, eos_index)
        if step + 1 == max_output_length:
            # TODO(alex-fabbri):  probably have to change here
            is_finished.fill(True)
        # end condition is whether the top beam is finished
        end_condition = np.equal(is_finished[:, 0], True)

        # save finished hypotheses
        if is_finished.any():
            predictions = np.reshape(alive_seq, (-1, size, alive_seq.shape[-1]))
            for i in range(is_finished.shape[0]):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill(1)
                finished_hyp = np.reshape(np.nonzero(is_finished[i]), -1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    # Check if the prediction has more than one EOS.
                    # If it has more than one EOS, it means that the
                    # prediction should have already been added to
                    # the hypotheses, so you don't have to add them again.
                    if np.nonzero(predictions[i, j, 1:] == eos_index)[0].size < 2:
                        # ignore start_token
                        # TODO(alex-fabbri): check 
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:])
                        )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = np.reshape(np.nonzero(np.equal(end_condition, False)), -1)
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            if len(non_finished) == 0:
                break

            # remove finished batches for the next step
            topk_log_probs = np.take(topk_log_probs, non_finished, axis=0)
            batch_index = np.take(batch_index, non_finished, axis=0)
            batch_offset = np.take(batch_offset, non_finished, axis=0)
            alive_seq = np.reshape(np.take(predictions, non_finished, axis=0), (-1, alive_seq.shape[-1]))

        # reorder indices
        select_indices = np.reshape(batch_index, -1)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps([r[0] for r in
                                        results["predictions"]],
                                       pad_value=pad_index)

    return final_outputs, None
