import string

import torch


def get_embeddings(model):
    """
    model: tranformer model

    returns word embeddings
    """

    return model.bert.embeddings.word_embeddings.weight


def get_embedding_layer(model):
    """
    model: tranformer model

    returns word embedding layer (torch.nn.Module)
    """

    return model.bert.embeddings


def get_word_embeddings(model, w_ids):
    """
    model: tranformer model
    w_ids: word ids

    returns word embeddings corresponding to the ids
    """
    return get_embeddings(model)[w_ids]


def get_word_gradients(model, w_ids):
    """
    model: tranformer model
    w_ids: word ids

    returns gradients of the words corresponding to the ids
    """
    embeddings = get_embeddings(model)
    if embeddings.grad is None:
        print("Embeddings have no gradients")
    return embeddings.grad[w_ids]


def summarize_attributions(attributions):
    # TODO: what does this function do?
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def normalize_attributions(score_vector):
    """
    normalize the score

    Parameters
    ----------
    score_vector : torch.cuda.FloatTensor
        shape: max_seq_len
    """
    denominator = torch.sum(abs(score_vector))
    return score_vector / denominator


def normalize_attributions_by_max(score_vector):
    """
    normalize the score

    Parameters
    ----------
    score_vector : torch.cuda.FloatTensor
        shape: max_seq_len
    """
    denominator = torch.max(abs(score_vector))
    return score_vector / denominator


def are_all_chars_punctuations(word):
    # function returns true if all the characters in the word are punctuations

    # check if any character is not a punctuation
    for w in word:
        if w not in string.punctuation:
            return False
    return True


def add_highlights(tokens, scores, logreg=False, topk=True):
    # add background highlights on tokens, based on the score
    # NOTE: works for score values between -1 and 1 (both inclusive)
    assert len(tokens) == len(scores)
    text = ""
    val = sorted([abs(k) for k in scores])[-int(0.1 * len(scores))]
    max_val = max([abs(k) for k in scores])
    special_quote_words = ["'s", "'ve", "'m", "'t", "'es"]
    start_punct = ["(", "[", "{"]

    stack_punct = {}
    stack_punct["'"] = False
    stack_punct['"'] = False
    print(tokens)
    if logreg:
        start = 0
        end = len(tokens)
    else:
        # Do not include CLS and SEP tokens
        start = 1
        end = len(tokens) - 1
    max_val = max([abs(k) for k in scores[start:end]])
    for idx in range(start, end):
        subword = False
        if idx != (len(tokens) - 1):
            if len(tokens[idx + 1]) != 0:
                if len(tokens[idx + 1]) >= 2 and tokens[idx + 1][:2] == "##":
                    subword = True
                if not (logreg):
                    if tokens[idx] + tokens[idx + 1] in special_quote_words:
                        subword = True
                    elif are_all_chars_punctuations(tokens[idx + 1]):
                        if idx < len(tokens) - 2:
                            if tokens[idx + 1] + tokens[idx + 2] in special_quote_words:
                                subword = True
                        if subword == False:
                            if tokens[idx + 1] in start_punct:
                                subword = False
                            elif tokens[idx + 1] in stack_punct:
                                subword = stack_punct[tokens[idx + 1]]
                                stack_punct[tokens[idx + 1]] = not (
                                    stack_punct[tokens[idx + 1]]
                                )
                                print(stack_punct[tokens[idx + 1]])
                            else:
                                subword = True
                    elif are_all_chars_punctuations(tokens[idx]):
                        if tokens[idx] in start_punct:
                            subword = True
                        elif tokens[idx] in stack_punct:
                            subword = stack_punct[tokens[idx]]
                        else:
                            subword = False

        token = tokens[idx].replace("#", "")
        if token == "<unk>":
            token = "UNK"
        if topk:
            if abs(scores[idx]) < val:
                scores[idx] = 0
        if max_val != 0:
            print("max val")
            print(max_val)
            scores[idx] = scores[idx] / max_val
        html_code = max(0, int(255 - (255 * abs(scores[idx]))))
        if scores[idx] >= 0:
            # positive score means yellow background
            text += f'<span class="importance-word" style="background-color: rgb(255, 255, {html_code})">'
        else:
            # negative score means red background
            text += f'<span class="importance-word" style="background-color: rgb(255, {html_code}, 255)">'
        if subword:
            text += token + "</span>"
        else:
            text += token + "</span> "

    text += "\n"
    text += "<br>"
    return text


def add_meta_data(html, prediction, prob):
    text = (
        "<p>Prediction: %s (with Confidence = %.1f%%). Following is the explanation</p>"
        % (prediction, prob * 100)
    )
    text += html
    return text


def add_no_explanation(prediction, prob):
    text = "<p>Prediction: %s (with Confidence = %.1f%%).</p>" % (
        prediction,
        prob * 100,
    )
    return text
