import os
import json
import anthropic
import datetime
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 불러옵니다.
load_dotenv()

# API 클라이언트를 초기화합니다.
try:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    print(f"Anthropic 클라이언트 초기화 실패: {e}")
    client = None

# =========================================================================
# 1. LLM 시스템 지침 (System Prompt) 정의
# =========================================================================
SYSTEM_PROMPT = """
당신은 '나를 위한 소비' 성향과 스트레스 관리의 연관성을 분석하는 전문적인 인공지능 데이터 구조화 시스템입니다.

규칙:
1.  **[가장 중요] 순수 JSON 객체 출력:** 당신의 최종 응답은 **마크다운 코드 블록(예: ```json)을 포함하지 않은** 순수한 JSON 객체(`{...}`)여야 합니다. 다른 설명, 주석, 마크다운 문법을 절대 추가하지 마십시오.
2.  **역할 및 분석:** 사용자의 자연어 질문('user_query')을 분석하여 심리 상태와 소비 의도를 구조화하십시오.
3.  **필수값 채우기:** 모든 필드를 최대한 채우십시오. 추출 불가능 시 STRING은 "", INTEGER는 0, BOOLEAN은 false를 사용하십시오.
4.  **표준화 강제 ('stress_factors'):** 'factor_name'은 다음 6가지 표준 항목 중 하나를 정확히 사용해야 합니다: "업무 / 학업", "출퇴근", "인간관계", "경제적 문제", "건강 문제", "기타".
5.  **메타데이터 처리:** 'model_name'은 "claude-sonnet-4-20250514", 'prompt_version'은 "V1.0"을 사용하십시오.
"""

# =========================================================================
# 2. JSON 스키마 출력 템플릿 로드 (경로 및 파일명 수정)
# =========================================================================
def load_json_schema(file_path: str) -> str:
    """JSON 파일을 읽어 문자열로 반환합니다."""
    # os.path.join을 사용하여 OS 독립적인 경로를 생성하는 것이 더 안전하지만, 
    # 여기서는 상대 경로를 명시적으로 사용합니다.
    try:
        # 파일 경로를 변경했습니다.
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"오류: 스키마 파일을 찾을 수 없습니다 - {file_path}")
        return "{}"

# 스크립트 실행 시점에 스키마를 로드합니다. (경로 수정 반영)
JSON_SCHEMA_TEMPLATE = load_json_schema("../json/poll10prompt_schema.json")


# =========================================================================
# 3. API 호출 함수 정의
# =========================================================================
def get_structured_response(user_input_text: str, request_id: str) -> str:
    """
    사용자 질문을 Claude 모델에 보내고 JSON 스키마 형식의 응답을 반환합니다.
    """
    if not client:
        return "Anthropic 클라이언트가 초기화되지 않았습니다."
    
    # 스키마 로드 실패 시 예외 처리
    if JSON_SCHEMA_TEMPLATE == "{}":
        return "JSON 스키마 템플릿을 불러오지 못하여 API 호출을 건너뜁니다."

    # 동적인 데이터 주입
    current_time = datetime.datetime.now().isoformat()

    # 스키마 템플릿에 동적 변수 주입
    filled_schema_template = JSON_SCHEMA_TEMPLATE.replace("{unique_request_id}", request_id)
    filled_schema_template = filled_schema_template.replace("{current_datetime}", current_time)
    filled_schema_template = filled_schema_template.replace("{user_input_text}", user_input_text)

    # LLM에 보낼 최종 사용자 프롬프트 구성
    final_user_prompt = f"""
    [입력 데이터]
    사용자 질문 (USER_QUERY): "{user_input_text}"

    [요청]
    위 '사용자 질문'을 분석하고, 아래의 JSON 스키마 템플릿에 맞추어 모든 필드를 채운 최종 JSON 객체만을 출력하십시오.

    [JSON 스키마 출력 템플릿]
    {filled_schema_template}
    """

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048, 
            system=SYSTEM_PROMPT, 
            messages=[
                {"role": "user", "content": final_user_prompt}
            ]
        ).content[0].text
        return message
    except Exception as e:
        return f"API 호출 중 오류 발생: {e}"

# =========================================================================
# 4. 테스트 코드 및 JSON 파싱 검증
# =========================================================================
if __name__ == '__main__':
    # -----------------------------------------------------------
    # Test Scenario 2: 복수 요인 및 정보 부족 처리 쿼리
    # -----------------------------------------------------------
    test_query = "계속된 야근이랑 지긋지긋한 출퇴근길 때문에 너무 지쳤어. 나를 위한 소비를 해야겠는데, 뭐 해야 할지 모르겠네. 돈은 별로 없어."
    test_id = "REQ_20251011_002" 
    
    print(f"--- Claude AI에 JSON 구조화 요청 --- (쿼리 ID: {test_id})")
    print(f"--- 쿼리: {test_query[:40]}... ---")
    
    response_json_string = get_structured_response(test_query, test_id)
    
    # LLM 응답 출력
    print("\n[Claude AI 응답 (Raw String)]")
    print(response_json_string)

    # 응답이 유효한 JSON인지 검증 (스키마 강제 성공 여부 확인)
    try:
        parsed_json = json.loads(response_json_string)
        print("\n--- JSON 파싱 성공 (스키마 강제 성공) ---")
        
        # 🌟 복수 요인 및 정보 부족에 대한 핵심 검증 🌟
        factors = [f['factor_name'] for f in parsed_json['stress_analysis']['stress_factors']]
        
        print(f"  > 분석된 스트레스 요인 수: {len(factors)}개")
        print(f"  > 분석된 스트레스 요인 목록: {factors}") # '업무 / 학업', '출퇴근' 포함 예상
        print(f"  > 분석된 예산 범위: {parsed_json['consumption_intention']['budget_range']['min']} ~ {parsed_json['consumption_intention']['budget_range']['max']} (예상: 0 ~ 0)")
        print(f"  > 분석된 카테고리: '{parsed_json['consumption_intention']['category_preference']}' (예상: 빈 문자열)")
        
    except json.JSONDecodeError:
        print("\n!!! JSON 파싱 실패 (스키마 강제 실패) - 시스템 지침 튜닝 필요 !!!")