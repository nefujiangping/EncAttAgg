import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from models.long_seq import process_long_input


class DocumentEncoder(nn.Module):

    def __init__(self, pretrained_model_name_or_path: str = 'bert-base-cased',
                 transformer_type: str = 'bert',
                 num_hidden_layers: int = 12,
                 freeze: bool = False):
        super(DocumentEncoder, self).__init__()
        print("\n\nDocumentEncoder: {}, {}\n".format(
            transformer_type, pretrained_model_name_or_path))
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path, mirror='tuna')
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, mirror='tuna')
        self.config.cls_token_id = tokenizer.cls_token_id
        self.config.sep_token_id = tokenizer.sep_token_id
        self.config.transformer_type = transformer_type
        self.config.num_hidden_layers = num_hidden_layers
        self.document_transformer_encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=self.config, mirror='tuna')
        for param in self.document_transformer_encoder.parameters():
            param.requires_grad = not freeze
        self.output_dim = self.config.hidden_size

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        else:
            raise NotImplementedError("bert or roberta.")
        sequence_output = process_long_input(self.document_transformer_encoder, input_ids, attention_mask, start_tokens, end_tokens)
        assert sequence_output.size(-1) == self.output_dim
        return sequence_output
