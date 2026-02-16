import asyncio
import json
import re
from typing import Dict, List, Set, Tuple, Union
from openai import AsyncOpenAI

class AdaptiveAgentWorkflow:
    """
    PCST가 정제한 서브그래프를 바탕으로, 세 명의 페르소나 에이전트가 협력하여
    최종 스키마를 확정하거나 불확실성(Uncertainty)을 근거로 답변을 거절(Unanswerable)합니다.
    """
    def __init__(self, 
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                 api_base: str = "http://localhost:8000/v1", 
                 api_key: str = "vllm",
                 uncertainty_threshold: float = 0.3):
        # 비동기 처리를 위한 AsyncOpenAI 클라이언트
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.threshold = uncertainty_threshold # 불확실성 임계값 (0.0 ~ 1.0)

    def _generate_ddl(self, subgraph: Dict[str, List[str]]) -> str:
        """PCST 결과를 LLM이 가장 이해하기 쉬운 DDL 형태로 변환합니다."""
        ddl_lines = []
        for table, columns in subgraph.items():
            cols_str = ",\n  ".join([f"{col} TEXT" for col in columns]) # 타입은 단순화
            ddl_lines.append(f"CREATE TABLE {table} (\n  {cols_str}\n);")
        return "\n\n".join(ddl_lines)

    def _extract_json(self, response_text: str) -> dict:
        """LLM 출력에서 마크다운(```json ... ```)을 제거하고 순수 JSON만 파싱합니다."""
        try:
            json_str = re.search(r'\{.*\}', response_text.replace('\n', ' '), re.DOTALL)
            if json_str:
                return json.loads(json_str.group())
            return json.loads(response_text)
        except json.JSONDecodeError:
            print(f"[Warning] JSON Parsing failed. Raw output: {response_text}")
            return {"step_by_step_reasoning": "Parse Error", "selected_nodes": []}

    async def _call_agent(self, role: str, prompt: str) -> dict:
        """비동기로 LLM을 호출하고 JSON을 파싱하여 반환합니다."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=256
            )
            return self._extract_json(response.choices[0].message.content)
        except Exception as e:
            print(f"Agent API Error: {e}")
            return {"selected_nodes": []}

    async def run_workflow(self, nl_query: str, subgraph: Dict[str, List[str]]) -> Dict[str, Union[float, str, List[str]]]:
        """[핵심] 3단계 Adaptive Workflow를 실행합니다."""
        
        ddl_schema = self._generate_ddl(subgraph)
        
        # ---------------------------------------------------------
        # Phase 1: 비동기 병렬 평가 (Semantic vs Structural)
        # ---------------------------------------------------------
        semantic_role = "You are a Semantic Data Analyst. Return strictly JSON: {'step_by_step_reasoning': '...', 'selected_nodes': ['table.col1', ...]}"
        semantic_prompt = f"Query: '{nl_query}'\nSchema:\n{ddl_schema}\nSelect ONLY the strictly necessary columns based on the semantic meaning of the user query."

        structural_role = "You are a Structural Database Administrator. Return strictly JSON: {'step_by_step_reasoning': '...', 'selected_nodes': ['table.col1', ...]}"
        structural_prompt = f"Query: '{nl_query}'\nSchema:\n{ddl_schema}\nEnsure valid JOIN paths exist. Select the strictly necessary tables and columns (including primary/foreign keys) to maintain SQL integrity."

        print("[Phase 1] Launching parallel agents (Semantic & Structural)...")
        # 두 에이전트를 동시에 실행 (시간 절약)
        semantic_res, structural_res = await asyncio.gather(
            self._call_agent(semantic_role, semantic_prompt),
            self._call_agent(structural_role, structural_prompt)
        )

        set_a = set(semantic_res.get("selected_nodes", []))
        set_b = set(structural_res.get("selected_nodes", []))

        # ---------------------------------------------------------
        # Phase 2: Jaccard 기반 불확실성(Uncertainty) 계산
        # ---------------------------------------------------------
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        # U = 1 - (교집합 / 합집합). 완벽히 일치하면 0, 완전히 다르면 1.
        uncertainty = 1.0 - (intersection / union) if union > 0 else 1.0
        print(f"[Phase 2] Consensus Check -> Uncertainty Score: {uncertainty:.2f}")

        # 합의(Consensus) 성공: 불확실성이 임계값보다 낮음
        if uncertainty < self.threshold:
            print("=> Consensus Reached. Bypassing Skeptic Agent.")
            return {
                "status": "Answerable",
                "uncertainty": uncertainty,
                "final_nodes": list(set_a.union(set_b)),
                "reasoning": "Agents reached consensus."
            }

        # ---------------------------------------------------------
        # Phase 3: 조건부 회의론자 개입 (The Conditional Skeptic)
        # ---------------------------------------------------------
        print(f"=> High Uncertainty Detected (> {self.threshold}). Triggering Skeptic Agent...")
        
        skeptic_role = "You are a Conservative Skeptic. Resolve conflicts between agents. If the query is ambiguous, missing information, or impossible with this schema, output 'Unanswerable'. Return strictly JSON: {'step_by_step_reasoning': '...', 'final_decision': ['table.col1', ...] OR 'Unanswerable'}"
        skeptic_prompt = f"""
Query: '{nl_query}'
Schema:\n{ddl_schema}

Agent A (Semantic) selected: {list(set_a)}
Reasoning: {semantic_res.get('step_by_step_reasoning', '')}

Agent B (Structural) selected: {list(set_b)}
Reasoning: {structural_res.get('step_by_step_reasoning', '')}

Analyze the conflict. Is this query genuinely answerable without making dangerous assumptions? 
If yes, provide the unified list of nodes. If no, strictly write "Unanswerable" in the final_decision field.
"""
        skeptic_res = await self._call_agent(skeptic_role, skeptic_prompt)
        decision = skeptic_res.get("final_decision", "Unanswerable")

        status = "Unanswerable" if decision == "Unanswerable" else "Answerable"
        final_nodes = [] if status == "Unanswerable" else decision

        return {
            "status": status,
            "uncertainty": uncertainty,
            "final_nodes": final_nodes,
            "reasoning": skeptic_res.get("step_by_step_reasoning", "")
        }

# --- 테스트 코드 (Unit Test) ---
async def main():
    # PCST 라우터가 넘겨준 정제된 서브그래프 (예시)
    mock_subgraph = {
        "employee": ["id", "name", "salary", "department_id"],
        "department": ["id", "name"]
    }
    nl_query = "What is the name of the department where the employee with the highest salary works?"

    # 워크플로우 초기화
    workflow = AdaptiveAgentWorkflow()

    # 실행
    result = await workflow.run_workflow(nl_query, mock_subgraph)
    
    print("\n--- Final Workflow Result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # 비동기 이벤트 루프 실행
    asyncio.run(main())