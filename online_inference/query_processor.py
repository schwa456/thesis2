import spacy
import torch
from typing import List, Tuple

class QueryProcessor:
    """
    사용자의 자연어 질의(NLQ)를 받아 핵심 토큰(명사, 고유명사 등)만 필터링하고,
    FAISS 탐색 시 노이즈를 줄이기 위한 마스킹(Masking)을 수행합니다.
    """
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        # SpaCy 모델 로드 (품사 태깅용)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spacy model {spacy_model}...")
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(spacy_model)
            
        # 유지할 핵심 품사 (명사, 고유명사, 형용사 등)
        self.valid_pos = {'NOUN', 'PROPN', 'ADJ'} 

    def extract_keywords(self, query: str) -> List[str]:
        """질문에서 불용어를 제외한 핵심 키워드 리스트를 반환합니다."""
        doc = self.nlp(query)
        keywords = [token.text for token in doc if token.pos_ in self.valid_pos and not token.is_stop]
        return keywords

    def mask_embeddings(self, 
                        token_embs: torch.Tensor, 
                        tokens: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        PLMEncoder에서 나온 전체 토큰 임베딩 중, 
        유효한 키워드(불용어/특수기호 제외)에 해당하는 행(Row)만 추출합니다.
        
        입력 token_embs shape: (Seq_Len, Hidden_Dim)
        """
        valid_indices = []
        valid_tokens = []
        
        # 문장 전체를 다시 합쳐서 SpaCy로 분석하여 품사 확인
        clean_text = " ".join([t.replace("##", "") for t in tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]])
        doc = self.nlp(clean_text)
        
        # SpaCy의 토큰과 Sub-word 토큰 매칭 (휴리스틱 간소화)
        valid_words = {t.text.lower() for t in doc if t.pos_ in self.valid_pos and not t.is_stop}
        
        for i, token in enumerate(tokens):
            clean_token = token.replace("##", "").lower()
            if clean_token in valid_words or clean_token.isalnum():
                # 특수기호나 CLS/SEP 토큰 제외
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    valid_indices.append(i)
                    valid_tokens.append(token)
                    
        if not valid_indices: # 극단적인 경우 (모든 단어가 잘림) 대비
            return token_embs, tokens
            
        masked_embs = token_embs[valid_indices] # (K, Hidden_Dim) 크기로 축소됨
        return masked_embs, valid_tokens

# --- 단위 테스트 ---
if __name__ == "__main__":
    processor = QueryProcessor()
    q = "What is the average salary of the employees in the IT department?"
    print(f"Original: {q}")
    print(f"Keywords: {processor.extract_keywords(q)}")
    # Expected: ['average', 'salary', 'employees', 'IT', 'department']pip