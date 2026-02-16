import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple

class PLMEncoder(nn.Module):
    """
    HuggingFace PLM을 사용하여 문장의 Token-Level Embedding을 추출합니다.
    (Sentence Transformer의 Mean Pooling을 우회하여 각 단어의 문맥 벡터를 확보)
    """
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        super(PLMEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        입력: 자연어 리스트 (예: User Query 또는 Edge Description)
        출력: (Batch, Seq_Len, Hidden_Dim) 크기의 Token 임베딩 텐서와 파싱된 토큰 리스트
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # last_hidden_state의 shape: (Batch_size, Sequence_length, Hidden_dim)
        token_embeddings = outputs.last_hidden_state
        
        # 텍스트로 된 토큰 리스트 복원 (디버깅 및 마스킹 용도)
        batch_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
        
        return token_embeddings, batch_tokens