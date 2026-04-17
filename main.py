import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

system_prompt = """
당신은 20년차 IT, AI 강의 경력과 강의력이 수준급인 강사이자, 교육회사 스타트업 대표입니다.
당신의 강의 철학은 '1시간이든 6개월이든 시간을 들이는 만큼 수강생들에게 만족감과 실제적인 도움(취업, 업무능력향상 등)을 주기 위한 강의를 하는 것'입니다.
이 철학을 바탕으로, 사용자(기업 교육 담당자 등)가 요청하는 조건에 맞춰 최적의 AX(AI Transformation) 과정 커리큘럼을 작성해야 합니다.

[요구사항 및 조건]
1. 대상자(수준, 직군 등)를 정확히 고려하여 난이도와 학습 방향성을 설계할 것.
2. 요구받은 기능 혹은 핵심 주제들이 빠짐없이 실무적 관점에서 커리큘럼에 반영될 것.
3. 구체적인 목차(시간/주차별), 학습 목표, 기대 효과, 그리고 강사로서의 짧은 코멘트(어떤 점을 중점적으로 가르칠 것인지)를 포함할 것.
4. 친절하고 자신감 있으며 전문가다운 어조로 답변할 것.
"""

def print_welcome_message():
    print("=" * 60)
    print("🤖 AX 커리큘럼 생성 챗봇에 오신 것을 환영합니다!")
    print("저는 20년차 IT/AI 강사이자 교육회사 스타트업 대표입니다.")
    print("수강생에게 실질적인 도움을 주는 맞춤형 커리큘럼을 만들어 드립니다.")
    print("=" * 60)
    print("시작하려면 교육 대상자와 주제를 먼저 알려주세요.")
    print("(언제든 'q' 또는 'quit'를 입력하여 종료할 수 있습니다)\n")

def chat_app():
    print_welcome_message()
    
    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 1. 초기 커리큘럼 생성
    target_audience = input("👤 교육 대상자 (예: 비전투입직군, 주니어 개발자): ")
    if target_audience.lower() in ['q', 'quit']: return
    
    topics = input("📋 교육 목적/주제 및 반영 기능 (예: ChatGPT 활용 데이터 분석): ")
    if topics.lower() in ['q', 'quit']: return
    
    user_prompt = f"아래 조건에 맞는 AX 교육 커리큘럼을 먼저 작성해 주세요.\n- 교육 대상자: {target_audience}\n- 핵심 주제: {topics}"
    conversation_history.append({"role": "user", "content": user_prompt})
    
    print("\n⏳ 커리큘럼을 고안하는 중입니다...\n")
    print("-" * 60)
    
    _generate_response(conversation_history)
    
    # 2. 후속 대화 루프
    while True:
        try:
            user_input = input("\n💬 강사에게 추가로 요청할 사항이나 수정할 부분이 있나요? (종료: q): ")
            if user_input.lower() in ['q', 'quit']:
                print("\n챗봇을 종료합니다. 수고하셨습니다!")
                break
                
            if not user_input.strip():
                continue
                
            conversation_history.append({"role": "user", "content": user_input})
            print("\n⏳ 답변을 생성 중입니다...\n")
            print("-" * 60)
            
            _generate_response(conversation_history)
                
        except (KeyboardInterrupt, EOFError):
            print("\n\n챗봇을 강제 종료합니다.")
            break

def _generate_response(messages):
    full_response = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # gpt-3.5-turbo 또는 gpt-4o 
            messages=messages,
            stream=True
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content
        print("\n" + "-" * 60)
        
        # 모델 응답을 히스토리에 추가
        messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        print(f"\n[오류 발생] OpenAI API 통신 중 문제가 발생했습니다: {e}")

if __name__ == "__main__":
    chat_app()
