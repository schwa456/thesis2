import os
import json
from typing import Dict, List, Any
from openai import OpenAI
from utils.logger import data_logger

class SchemaVerbalizer:
    """
    파싱된 스키마의 Foreign Key(엣지) 정보를 입력받아,
    Open Source LLM (vLLM, Ollama 등)을 활용해 자연어 설명(Description)을 생성합니다.
    """
    def __init__(self, model_name: str = "llama3", api_base: str = "http://localhost:11434/v1", api_key: str = "vllm"):
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key
        )

        self.model_name = model_name
    
    def _build_prompt(self, fk: Dict[str, str]) -> str:
        """
        LLM이 엣지의 의미를 정확히 파악할 수 있도록 프롬프트를 구성합니다.
        (논문 기여점: 사용자가 질문할 법한 '의미적 연결성'을 강조하도록 유도)
        """
        from_table = fk['from_table']
        from_col = fk['from_column']
        to_table = fk['to_table']
        to_col = fk['to_column']

        prompt = f"""
        You are a database expert. Describe the semantic relationship between two tables based on their foreign key connection.
        Keep it to a single, concise sentence that highlights the business logic or real-world connection a user might ask about.

        Connection Details:
        - The '{from_table}' table connects to the '{to_table}' table.
        - The foreign key '{from_col}' in '{from_table}' references '{to_col}' in '{to_table}'.

        Description:
        """

        return prompt.strip()
    
    def verbalize_foreign_key(self, fk: Dict[str, str]) -> str:
        """단일 FK에 대한 자연어 설명을 LLM에 요청하여 반환합니다."""
        prompt = self._build_prompt(fk)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful data dictionary assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            description = response.choices[0].message.content.strip()
            return description
        
        except Exception as e:
            data_logger.error(f"[ERROR] Error verbalizing FK {fk}: {e}")
            return f"Connects {fk['from_table']} to {fk['to_table']}."
    
    def process_all_fks(self, schema_info: Dict[str, Any]) -> Dict[str, str]:
        """
        스키마 내의 모든 FK를 순회하며 Edge Description 딕셔너리를 생성합니다.
        Key는 'from_table.from_col->to_table.to_col' 형태의 Edge ID입니다.
        """
        edge_descriptions = {}
        foreign_keys = schema_info.get("foreign_keys", [])

        data_logger.debug(f"Starting verbalization for {len(foreign_keys)} foreign keys...")
        
        for fk in foreign_keys:
            # Edge 고유 ID 생성 (예: employee.dept_id->department.dept_id)
            edge_id = f"{fk['from_table']}.{fk['from_column']}->{fk['to_table']}.{fk['to_column']}"
            
            description = self.verbalize_foreign_key(fk)
            edge_descriptions[edge_id] = description
            data_logger.debug(f"[*] {edge_id} : {description}")

        return edge_descriptions