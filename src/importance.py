import lime
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from lime.lime_text import LimeTextExplainer
from torch.autograd import Variable
from tqdm import tqdm

import importance_utils

"""Methods for getting importance scores"""


class AttentionExplainer:
    def __init__(self):
        pass

    def get_scores(self, model, database, user_name):
        """
        get attention scores.
        These scores are averaged values of attention originating from [CLS] token
        in the last layer averaged over all heads

        Parameters
        ----------
        model : torch.nn.Module
            bert-like model
        database : database for each user
        user_name : user_name of user

        Returns
        -------
        list[list]
            normalized attention scores, a list corresponding to each sentence
        """
        batch_size = database[user_name].input_ids.shape[0]
        output = model(
            database[user_name].input_ids, database[user_name].attention_mask
        )
        # output is a two tuple with first item as logits, and second item as attention scores
        attention = output[1]
        # for each layer attention is (batch_size, num_heads, sequence_length, sequence_length)
        last_layer_attention = attention[-1]

        # aggregate all the attention heads by mean
        last_layer_mean_attention = torch.mean(last_layer_attention, dim=1)
        # last_layer_mean_attention is (batch_size, sequence_length, sequence_length)

        last_layer_mean_CLS_attention = last_layer_mean_attention[:, 0, :]
        # last_layer_mean_CLS_attention is (batch_size, sequence_length)

        # NOTE: at this point converting to lists also in not a big time burden

        scores = []

        for i in range(batch_size):
            line_len = len(
                database[user_name].input_ids[i].nonzero()
            )  # including [CLS] and [SEP]
            # shape of score is line_len
            score = last_layer_mean_CLS_attention[i][:line_len]
            score = importance_utils.normalize_attributions_by_max(score)
            scores.append(score.tolist())

        return scores


class IntegratedGradientExplainer:
    def __init__(self):
        pass

    def get_scores(self, model, database, session):
        """
        get Integrated Gradient scores.

        Parameters
        ----------
        model : torch.nn.Module
            bert-like model
        database : database for each user
        user_name : user_name of user

        Returns
        -------
        list[list]
            normalized integrated gradients, a list corresponding to each sentence
        """
        self.model = model
        print(
            session.get("name")
            + " Input Id "
            + str(database[session.get("name")].input_ids.shape)
        )
        print(
            session.get("name")
            + " Attention Mask "
            + str(database[session.get("name")].attention_mask.shape)
        )
        output = self.model(
            database[session.get("name")].input_ids,
            database[session.get("name")].attention_mask,
        )
        predictions = torch.argmax(output[0], dim=-1)
        print(predictions)
        if hasattr(model, "model"):
            self.bert_model = self.model.model.bert
        else:
            self.bert_model = self.model.bert
        lig = LayerIntegratedGradients(self.model, self.bert_model.embeddings)
        scores = []
        for i, prediction in enumerate(predictions):
            line_len = len(
                database[session.get("name")].input_ids[i].nonzero()
            )  # including [CLS] and [SEP]
            database[session.get("name")].input_example = (
                database[session.get("name")].input_ids[i].unsqueeze(dim=0)
            )
            database[session.get("name")].baseline_example = torch.zeros_like(
                database[session.get("name")].input_example
            )  # all padded to zeros
            database[session.get("name")].attention_mask_example = (
                database[session.get("name")].attention_mask[i].unsqueeze(dim=0)
            )
            only_preds = torch.ones(1).bool()

            attributions = lig.attribute(
                inputs=(
                    database[session.get("name")].input_example,
                    database[session.get("name")].attention_mask_example,
                    only_preds,
                ),
                baselines=(
                    database[session.get("name")].baseline_example,
                    database[session.get("name")].attention_mask_example,
                    only_preds,
                ),
                target=prediction.item(),
                internal_batch_size=1,
            )

            score = importance_utils.summarize_attributions(attributions)
            score = importance_utils.normalize_attributions_by_max(score[:line_len])
            scores.append(score.tolist()[:line_len])
        print(scores)
        return scores


class GradientBasedImportanceScores:
    def __init__(self, method):
        # method can be "grad_norm" or "grad_times_inp" corresponding to two popular
        # attribution methods
        self.method = method

    def get_grad_norm(self, gradients):
        """gradient norm"""
        # grads shape is : seq_len x dim_size
        return torch.norm(gradients, p=2, dim=1)

    def get_grad_times_inp(self, gradients, w_embs):
        """gradient x input"""
        # grads shape is : seq_len x dim_size
        # w_embs shape is : seq_len x dim_size
        return torch.einsum("ij,ij->i", gradients, w_embs)

    def get_scores(self, model, input_ids, attention_mask):
        """
        get gradient-based importance scores.

        Parameters
        ----------
        model : torch.nn.Module
            bert-like model
        input_ids : torch.cuda.LongTensor
            input ids corresponding to tokens.
            shape: batch_size x max_seq_len
        attention_mask : torch.cuda.LongTensor
            shape: batch_size x max_seq_len
            value of 1 denotes that token participates in self-attention, 0 implies otherwise

        Returns
        -------
        list[list]
            normalized gradient-based importance scores, a list corresponding to each sentence
        """
        output = model(input_ids, attention_mask)
        # output is a two tuple with first item as logits, and second item as attention scores
        predictions = torch.argmax(output[0], dim=-1)
        scores = []  # list of list
        for i, prediction in enumerate(predictions):
            line_len = len(input_ids[i].nonzero())  # including [CLS] and [SEP]
            token_ids = input_ids[i][:line_len]

            log_prob_predicted = torch.log(
                torch.nn.functional.softmax(output[0][i], dim=0)
            )[prediction.item()]
            model.zero_grad()
            log_prob_predicted.backward(retain_graph=True)
            # check if the model is a wrapper for temp scaling or not
            if hasattr(model, "model"):
                bert_model = model.model
            else:
                bert_model = model
            grads = importance_utils.get_word_gradients(bert_model, token_ids)
            w_embs = importance_utils.get_word_embeddings(bert_model, token_ids)

            if "grad_norm" in self.method:
                score = self.get_grad_norm(grads)
            elif "grad_times_inp" in self.method:
                score = self.get_grad_times_inp(grads, w_embs)
            else:
                raise ValueError("method has to be grad_norm or grad)times_inp")

            score = importance_utils.normalize_attributions_by_max(score)
            scores.append(score.tolist())

        return scores


class LimeExplainer:
    def __init__(self):
        self.cuda = torch.cuda.is_available()

    def get_html_explanations(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        class_names=["Fake", "Genuine"],
    ):
        self.model = model
        self.tokenizer = tokenizer
        output = model(input_ids, attention_mask)
        predictions = torch.argmax(output[0], dim=-1)
        explainer = LimeTextExplainer(class_names=class_names)
        # the class names don't really matter actually
        output_html_explanations = []
        for i, prediction in enumerate(predictions):
            line_len = len(input_ids[i].nonzero())  # including [CLS] and [SEP]
            token_ids = input_ids[i][:line_len]

            text = " ".join(self.tokenizer.convert_ids_to_tokens(token_ids))

            num_features = min(10, line_len)
            # selecting 2 times features as some features would be for the opposite class

            exp = explainer.explain_instance(
                text,
                self.predict_prob_batch,
                num_features=num_features,
                num_samples=2 * line_len,
            )
            # at least two times as many samples so that LIME has sufficient chance of replacing
            # every token at least a few times.
            # for k in exp.asmap()[]
            print(exp.as_list())

            output_html_explanations.append(exp.as_list())

        return output_html_explanations

    def get_token_level_score(self, score_list, tokens):
        score_dict = {}
        max_weight = -1
        for (word, weight) in score_list:
            score_dict[word] = weight
            if abs(weight) > max_weight:
                max_weight = abs(weight)

        scores = []
        for tok in tokens:
            if tok not in score_dict:
                scores.append(0)
            else:
                scores.append(score_dict[tok] / max_weight)
        return scores

    def predict_prob(self, text):
        tokens = text.split()  # self.tokenizer.tokenize(text)
        tokens = tokens[:512]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids])
        if self.cuda:
            input_ids = input_ids.type(torch.cuda.LongTensor)
        attention_mask = torch.ones_like(input_ids)
        output = self.model(input_ids, attention_mask)
        output = torch.nn.functional.softmax(output[0], dim=1)[0]
        return output.detach().cpu().numpy()

    def predict_prob_batch(self, inputs):
        output = [self.predict_prob(i) for i in inputs]
        return np.array(output)
