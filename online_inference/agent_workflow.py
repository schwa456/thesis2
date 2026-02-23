import asyncio
import json
import ast
import re
from typing import Dict, List, Set, Tuple, Union
from openai import AsyncOpenAI
from utils.prompts import SEMANTIC_SYSTEM_ROLE, SEMANTIC_USER_PROMPT, STRUCTURAL_SYSTEM_ROLE, STRUCTURAL_USER_PROMPT, SKEPTIC_SYSTEM_ROLE, SKEPTIC_USER_PROMPT
from utils.logger import agent_logger

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
        """LLM 출력에서 JSON(또는 Dict) 포맷을 강제로 추출하여 파싱합니다."""
        # 1. 텍스트에서 {} 형태의 블록만 추출
        match = re.search(r'\{.*\}', response_text.replace('\n', ' '), re.DOTALL)
        json_str = match.group() if match else response_text

        try:
            # 2. 먼저 표준 JSON 파싱 시도 (쌍따옴표)
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # 3. 실패 시 파이썬 ast.literal_eval을 사용해 홑따옴표 문자열 파싱 시도
                return ast.literal_eval(json_str)
            except Exception as e:
                agent_logger.warning(f"[Warning] JSON Parsing failed. Raw output:\n{response_text}")
                # 기본 거절(Unanswerable) 포맷 반환으로 시스템 다운 방지
                return {"step_by_step_reasoning": "Parse Error", "selected_nodes": [], "final_decision": "Unanswerable"}

    async def _call_agent(self, name: str, role: str, prompt: str) -> dict:
        """비동기로 LLM을 호출하고 JSON을 파싱하여 반환합니다."""
        try:
            # 시스템 프롬프트에 중복 방지와 JSON 마무리를 강력히 지시
            enhanced_role = role + (
                " You MUST return ONLY a valid JSON object. "
                "DO NOT write any SQL queries. DO NOT use markdown formatting like ```json or ```sql. "
                "Do not add any conversational text. Start directly with { and end with }."
            )
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                frequency_penalty=0.5,
                response_format={"type": "json_object"}
            )
            output = self._extract_json(response.choices[0].message.content)
            agent_logger.debug(f"{name} Agent Reasoning:\n{output.get("step_by_step_reasoning", "")}")
            agent_logger.debug(f"{name} Agent Selected Nodes:\n{output.get("selected_nodes", [])}")
            agent_logger.debug(f"{name} Agent Final Decision: {output.get("final_decision", "")}")
            return output
        except Exception as e:
            agent_logger.error(f"Agent API Error: {e}")
            return {"step_by_step_reasoning": "API Error", "selected_nodes": [], "final_decision": "Unanswerable"}

    async def run_workflow(self, nl_query: str, subgraph: Dict[str, List[str]]) -> Dict[str, Union[float, str, List[str]]]:
        """[핵심] 3단계 Adaptive Workflow를 실행합니다."""
        agent_logger.debug("==========================================================================")
        agent_logger.debug(f"[Agent] Question: {nl_query}")

        ddl_schema = self._generate_ddl(subgraph)
        
        # ---------------------------------------------------------
        # Phase 1: 비동기 병렬 평가 (Semantic vs Structural)
        # ---------------------------------------------------------
        semantic_role = 'You are a Semantic Data Analyst. Return strictly JSON: {"step_by_step_reasoning": "...", "selected_nodes": ["table.col1", ...]}'
        semantic_prompt = SEMANTIC_USER_PROMPT.format(query=nl_query, schema_ddl=ddl_schema)

        structural_role = 'You are a Structural Database Administrator. Return strictly JSON: {"step_by_step_reasoning": "...", "selected_nodes": ["table.col1", ...]}'
        structural_prompt = STRUCTURAL_USER_PROMPT.format(query=nl_query, schema_ddl=ddl_schema)

        agent_logger.debug("[Phase 1] Launching parallel agents (Semantic & Structural)...")
        # 두 에이전트를 동시에 실행 (시간 절약)
        semantic_res, structural_res = await asyncio.gather(
            self._call_agent(name="Semantic Data Analyst", role=semantic_role, prompt=semantic_prompt),
            self._call_agent(name="Structural Database Administrator", role=structural_role, prompt=structural_prompt)
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
        agent_logger.debug(f"[Phase 2] Consensus Check -> Uncertainty Score: {uncertainty:.2f}")

        # 합의(Consensus) 성공: 불확실성이 임계값보다 낮음
        if uncertainty < self.threshold:
            agent_logger.debug("=> Consensus Reached. Bypassing Skeptic Agent.")
            return {
                "status": "Answerable",
                "uncertainty": uncertainty,
                "final_nodes": list(set_a.union(set_b)),
                "reasoning": "Agents reached consensus."
            }

        # ---------------------------------------------------------
        # Phase 3: 조건부 회의론자 개입 (The Conditional Skeptic)
        # ---------------------------------------------------------
        agent_logger.debug(f"=> High Uncertainty Detected (> {self.threshold}). Triggering Skeptic Agent...")
        
        skeptic_role = (
            "You are a Conservative Skeptic. Resolve conflicts between agents. "
            "If the query is ambiguous, missing information, or impossible with this schema, output the exact string \"Unanswerable\". "
            "Return strictly a JSON object with two keys: 'step_by_step_reasoning' (string) and 'final_decision' (either a list of strings like [\"table.col\", ...] OR the exact string \"Unanswerable\")."
        )
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
        skeptic_res = await self._call_agent(name="Conservative Skeptic", role=skeptic_role, prompt=skeptic_prompt)
        decision = skeptic_res.get("final_decision", "Unanswerable")

        if isinstance(decision, dict) and "Unanswerable" in decision:
            decision = "Unanswerable"

        status = "Unanswerable" if decision == "Unanswerable" else "Answerable"
        final_nodes = [] if status == "Unanswerable" else decision

        agent_logger.debug("==========================================================================")

        return {
            "status": status,
            "uncertainty": uncertainty,
            "final_nodes": final_nodes,
            "reasoning": skeptic_res.get("step_by_step_reasoning", "")
        }