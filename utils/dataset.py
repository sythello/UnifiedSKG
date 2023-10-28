import os
import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, ):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

        self.conv_sep = " || "

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]

        if raw_item["text_in"]:
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                ##################
                # Conversational #
                ##################
                # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                index = raw_item["text_in"].index(self.conv_sep)
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                  raw_item["struct_in"],
                                                                                  raw_item["text_in"][index + len(self.conv_sep):])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                       raw_item["text_in"][index + len(self.conv_sep):])
                else:
                    raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "text_in ; structured knowledge: struct_in"
                    seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "text_in"
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "structured knowledge: struct_in"
                seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()

        # Concatenate description.
        if self.args.model.use_description and self.args.model.concatenate_description:
            seq_in = "{} ; {}".format(raw_item["description"], seq_in)


        # YS NOTE: allow decoder-only (causalLM) model by concatenating input & output, and set labels to ignore input 
        if getattr(self.tokenizer, 'pad_token_id') is None:
            print(f'** (YS) Tokenizer has no `pad_token_id`, using eos_token_id = {self.tokenizer.eos_token_id} ({self.tokenizer.eos_token})')
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if getattr(self.args.model, 'is_causal_lm', False):
            # YS added for causal LMs.
            # Different from T5, GPT2 uses absolute positional embedding so we can't surpass the size (in the config)
            _connector = '; SQL:'
            _connector_len = len(self.tokenizer.tokenize(_connector))

            # # If output too long (unusual), truncate to half input length
            # seq_out = raw_item["seq_out"]
            # tokenized_inferred_toks = self.tokenizer.tokenize(seq_out)
            # tokenized_inferred_length = min(len(tokenized_inferred_toks), self.training_args.input_max_length // 2)
            # tokenized_inferred_toks = tokenized_inferred_toks[:tokenized_inferred_length]
            
            # tokenized_seq_in_toks = self.tokenizer.tokenize(seq_in)
            # # -2 to have some flexibility in space
            # tokenized_seq_in_length = min(len(tokenized_seq_in_toks), \
            #                               self.training_args.input_max_length - tokenized_inferred_length - _connector_len - 2)
            # tokenized_seq_in_toks = tokenized_seq_in_toks[:tokenized_seq_in_length]

            # # try:
            # rebuild_seq_out = self.tokenizer.convert_tokens_to_string(tokenized_inferred_toks)
            # if not rebuild_seq_out.endswith(';'):
            #     rebuild_seq_out = rebuild_seq_out + ';'     # Try to let the model learn to stop

            # rebuild_seq_in = self.tokenizer.convert_tokens_to_string(tokenized_seq_in_toks)
            # # rebuild_full_input = self.tokenizer.convert_tokens_to_string(tokenized_seq_in_toks + tokenized_inferred_toks)
            # rebuild_full_input = f'{rebuild_seq_in} {_connector} {rebuild_seq_out}'
            # # except:
            # #     breakpoint()

            # tokenized_full_input = self.tokenizer(
            #     rebuild_full_input,
            #     padding="max_length",
            #     max_length=self.training_args.input_max_length,
            #     # return_tensors="pt",  # Need to create separate tensors later, so not asking for tensor output here
            # )

            # # input_ids, labels: (1, seq_len)
            # labels = torch.LongTensor(tokenized_full_input.data["input_ids"])
            # labels[labels == self.tokenizer.pad_token_id] = -100
            # labels[:tokenized_seq_in_length + _connector_len] = -100     # YS: do not train on seq_in (and connector)

            # # YS: add "predict_input_ids" and "predict_attention_mask" here for predicting
            # rebuild_predict_input = f'{rebuild_seq_in} {_connector}'
            # tokenized_predict_input = self.tokenizer(
            #     rebuild_predict_input,
            #     padding=False,
            # )

            _padding_side = self.tokenizer.padding_side

            # Input
            # This pre-truncation process guarantees the connector to not be deleted (not sure if necessary though)
            tokenized_seq_in_toks = self.tokenizer.tokenize(seq_in)
            # -2 to have some flexibility in space
            tokenized_seq_in_length = min(len(tokenized_seq_in_toks), self.training_args.input_max_length - _connector_len - 2)
            tokenized_seq_in_toks = tokenized_seq_in_toks[:tokenized_seq_in_length]
            rebuild_seq_in = self.tokenizer.convert_tokens_to_string(tokenized_seq_in_toks)
            rebuild_input = f'{rebuild_seq_in} {_connector}'

            self.tokenizer.padding_side = 'left'
            tokenized_question_and_schemas = self.tokenizer(
                rebuild_input,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length,
            )

            # Output
            seq_out = ' ' + raw_item["seq_out"]     # prepend space to simulate concatenation
            seq_out_len = self.training_args.generation_max_length - self.training_args.input_max_length
            # seq_out_len = self.training_args.generation_max_length

            self.tokenizer.padding_side = 'right'
            tokenized_inferred = self.tokenizer(
                seq_out,
                padding="max_length",
                truncation=True,
                max_length=seq_out_len,
            )

            concat_input_ids = tokenized_question_and_schemas.data["input_ids"] + tokenized_inferred.data["input_ids"]
            concat_attention_mask = tokenized_question_and_schemas.data["attention_mask"] + tokenized_inferred.data["attention_mask"]
            labels = [-100] * self.training_args.input_max_length + tokenized_inferred.data["input_ids"]
            labels = torch.LongTensor(labels)
            labels[labels == self.tokenizer.pad_token_id] = -100

            item = {
                'input_ids': torch.LongTensor(concat_input_ids),
                'attention_mask': torch.LongTensor(concat_attention_mask),
                'predict_input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                'predict_attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                'labels': labels,
            }

            # print(seq_out)

            # (Restore settings)
            self.tokenizer.padding_side = _padding_side

        else:
            # original code for actual enc-dec models
            tokenized_question_and_schemas = self.tokenizer(
                seq_in,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length,
                # We found that set it as large as possible can boost the performance significantly
                # , meanwhile, due to the t5 uses a relative position coding, we need to manually
                # assign the max input length into some large numbers, instead of using the "max_model_length"
                # ,which the default is 512, which will hurt the performance a lot.
            )
            tokenized_inferred = self.tokenizer(
                raw_item["seq_out"],
                padding="max_length",
                truncation=True,
                max_length=self.training_args.generation_max_length,
                # We set the max_length of "seq_out" during training is the same with the one in inference.
            )

            tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
            # Here -100 will let the model not to compute the loss of the padding tokens.
            tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

            item = {
                'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                'labels': tokenized_inferred_input_ids,
            }
        
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.input_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])
        
        # print(item)
        # breakpoint()

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)
