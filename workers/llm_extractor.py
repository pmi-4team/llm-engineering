#
# 📌 파일 경로: workers/llm_extractor.py
# 📌 역할: Task A - DB 텍스트(title, body)를 받아 LLM으로 정규화/요약하여
# 📌         임베딩하기 좋은 텍스트(normalized_text)를 생성합니다.
#
import os
import json
import anthropic
import time # (추가) 재시도를 위한 time 모듈
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# API 클라이언트를 초기화합니다.
try:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # (MODIFIED) Task B에서 성공했던 모델 이름으로 변경합니다.
    MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514") # <- 여기를 수정!
    LLM_VERSION_BASE = f"{MODEL}-normalize-v0.1"
except Exception as e:
    print(f"Anthropic 클라이언트 초기화 실패: {e}")
    client = None

# =========================================================================
# [cite_start]1. LLM 시스템 지침 (System Prompt) 정의 (1) 누가...pdf [cite: 215-225] 참조)
# =========================================================================
SYSTEM_PROMPT = """너는 데이터 정규화/표준화를 수행하는 어시스턴트야.
- 노이즈/광고/이모지 제거
- 중복/장황함 축약
- 핵심 문장 유지 (1~3 문단으로 요약)
- 문장 부호/띄어쓰기 교정
- 금칙어/PII 제거 또는 마스킹 (필요시)
출력은 반드시 JSON 하나만 반환해. 다른 설명은 절대 금지.
{
  "normalized_text": "... 임베딩하기 좋은 한국어 1~3문단 요약 ...",
  "llm_version": "<모델명>-normalize-<버전>"
}
"""

# =========================================================================
# [cite_start]2. 반환 타입 정의 (1) 누가...pdf [cite: 177-183] 참조)
#    (참고: structured, evidence_spans는 Task A에서는 필수 아님)
# =========================================================================
class LLMNormalized(TypedDict):
    """LLM 정규화 결과 반환 타입"""
    normalized_text: str
    llm_version: str

# =========================================================================
# [cite_start]3. LLM 호출 함수 정의 (1) 누가...pdf [cite: 184-267] 참조)
# =========================================================================
def extract_normalized(title: str, body: str) -> LLMNormalized:
    """
    DB의 title과 body를 받아 LLM으로 정규화된 텍스트와 버전을 반환합니다.
    실패 시 원본 텍스트와 에러 버전으로 폴백(Fallback)합니다.
    """
    if not client:
        # 클라이언트 초기화 실패 시 즉시 폴백
        print("LLM 클라이언트 없음. 원본 텍스트로 폴백합니다.")
        return {
            "normalized_text": f"{title}\n{body}",
            "llm_version": f"{LLM_VERSION_BASE}-client-error"
        }

    prompt = f"""[TITLE]
{title}
[BODY]
{body}

[요구사항]
- 위 원문을 임베딩하기 좋게 한국어로 1~3문단으로 정리 (중복/광고 제거, 핵심만 남김)
- JSON만 출력 (설명 금지)
- llm_version 필드에는 "{LLM_VERSION_BASE}" 값을 넣어줘.
"""

    # (수정) [cite_start]API 호출 시 재시도 로직 추가 (1) 누가...pdf [cite: 235-258] 참조)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=1024, # (수정) 토큰 제한은 필요에 따라 조절
                temperature=0.1, # (수정) 낮은 온도로 일관성 유지
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                timeout=30, # (수정) 타임아웃 설정 (초)
            )
            content = message.content[0].text if message.content else "{}"

            # (수정) JSON 파싱 시도
            data = json.loads(content)

            # LLM이 normalized_text를 생성하지 못한 경우 폴백(Fallback)
            normalized = data.get("normalized_text")
            version = data.get("llm_version")

            # 필수 필드 누락 시 예외 발생시켜 재시도 유도 또는 폴백 처리
            if not normalized or not version:
                raise ValueError("LLM 응답에 필수 필드(normalized_text, llm_version) 누락")

            # 성공 시 결과 반환
            return {
                "normalized_text": normalized,
                "llm_version": version
            }

        except (anthropic.APIError, json.JSONDecodeError, ValueError) as e:
            print(f"LLM 정규화 시도 {attempt + 1}/{max_retries} 실패 (ID: {title[:20]}...): {e}.")
            if attempt < max_retries - 1:
                time.sleep(1 + attempt) # 재시도 전 잠시 대기
            else:
                # 최종 실패 시 폴백
                print("최대 재시도 실패. 원본 텍스트로 폴백합니다.")
                return {
                    "normalized_text": f"{title}\n{body}",
                    "llm_version": f"{LLM_VERSION_BASE}-fallback-error"
                }
        except Exception as e: # 예상치 못한 다른 에러 처리
             print(f"LLM 정규화 중 예상치 못한 오류 발생: {e}. 원본 텍스트로 폴백합니다.")
             return {
                "normalized_text": f"{title}\n{body}",
                "llm_version": f"{LLM_VERSION_BASE}-unexpected-error"
            }

# =========================================================================
# 4. (선택적) 테스트 코드
# =========================================================================
if __name__ == '__main__':
    # 테스트용 데이터
    test_title = "   ***광고*** 스트레스 확 풀리는 초특가 여행!! ✈️✈️"
    test_body = "힘든 일상, 훌쩍 떠나세요!\n\n최저가 보장! 지금 바로 예약하세요!\n\n문의: 1588-XXXX #여행 #스트레스 #특가 \n\n정말 힘들 땐 여행이 최고죠. 저도 지난주에 다녀왔는데 너무 좋았어요. 강추!!"

    print("--- LLM 텍스트 정규화 테스트 (Task A) ---")
    result = extract_normalized(test_title, test_body)

    print("\n[정규화 결과]")
    print(f"LLM Version: {result['llm_version']}")
    print(f"Normalized Text:\n{result['normalized_text']}")