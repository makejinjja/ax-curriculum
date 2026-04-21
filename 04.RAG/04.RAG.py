import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# 상단에서 바로 상대 경로를 통해 기존 .env 환경변수를 로드합니다.
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_ax_curriculum_chatbot', '.env'))
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

def setup_rag():
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'AXCompass.pdf'))
    if not os.path.exists(pdf_path):
        print(f"Error: 데이터 파일을 찾을 수 없습니다. 경로 확인 바랍니다: {pdf_path}")
        sys.exit(1)
        
    print("⏳ PDF 문서를 분석하고 벡터 공간을 구성하는 중입니다...")
    # PDF 로드
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # 텍스트 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    
    # VectorDB 에 저장
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ RAG 구성 완료!\n")
    return retriever

def get_context(retriever, query):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

system_prompt_template = """
당신은 20년차 IT, AI 강의 경력과 강의력이 수준급인 강사이자, 교육회사 스타트업 대표입니다.
당신의 강의 철학은 '1시간이든 6개월이든 시간을 들이는 만큼 수강생들에게 만족감과 실제적인 도움(취업, 업무능력향상 등)을 주기 위한 강의를 하는 것'입니다.
이 철학을 바탕으로, 사용자가 요청하는 조건에 맞춰 최적의 AX(AI Transformation) 과정 커리큘럼을 작성해야 합니다.

[요구사항 및 조건]
1. 대상자(수준, 직군 등)를 정확히 고려하여 난이도와 학습 방향성을 설계할 것.
2. 요구받은 기능 혹은 핵심 주제들이 빠짐없이 실무적 관점에서 커리큘럼에 반영될 것.
3. 구체적인 목차(시간/주차별), 학습 목표, 기대 효과, 그리고 강사로서의 짧은 코멘트(어떤 점을 중점적으로 가르칠 것인지)를 포함할 것.
4. [중요] **이론 수업은 모든 그룹이 동일하게 진행되도록 통합 교육으로 구성할 것.**
5. [중요] **실습/프로젝트는 진단 유형별로 만들어진 3개의 그룹으로 나누어, 각 그룹의 특성에 맞게 개별적으로 맞춤형 진행이 가능하도록 분리 구성할 것.**
   - 그룹 1: 균형형, 이해형
   - 그룹 2: 과신형, 실행형
   - 그룹 3: 판단형, 조심형
   (아래 첨부된 PDF 본문 발췌 컨텍스트(Context)를 적극 참고하여, 각 그룹 성향에 알맞은 실습 방식을 구체적으로 제안하세요.)
6. 친절하고 자신감 있으며 전문가다운 어조로 답변할 것.
"""

def print_welcome_message():
    print("=" * 70)
    print("🤖 AX 커리큘럼 생성 챗봇에 오신 것을 환영합니다! (RAG 성향 그룹화 맞춤 적용)")
    print("=" * 70)
    print("시작하려면 교육 대상자와 주제, 그리고 각 진단 유형별 인원수를 입력해주세요.")
    print("(언제든 'q' 또는 'quit'를 입력하여 종료할 수 있습니다)\n")

def chat_app():
    print_welcome_message()
    
    retriever = setup_rag()
    
    # 1. 초기 커리큘럼 생성
    target_audience = input("👤 교육 대상자 (예: 비전투입직군, 주니어 개발자): ")
    if target_audience.lower() in ['q', 'quit']: return
    
    topics = input("📋 교육 목적/주제 및 반영 기능 (예: ChatGPT 활용 데이터 분석): ")
    if topics.lower() in ['q', 'quit']: return

    print("\n📊 역량 진단 검사 결과의 6가지 유형별 인원수를 입력해주세요.")
    try:
        t_balanced = int(input(" - 균형형 인원수: "))
        t_understanding = int(input(" - 이해형 인원수: "))
        t_overconfident = int(input(" - 과신형 인원수: "))
        t_executing = int(input(" - 실행형 인원수: "))
        t_judging = int(input(" - 판단형 인원수: "))
        t_cautious = int(input(" - 조심형 인원수: "))
    except ValueError:
        print("\n[오류] 숫자를 올바르게 입력해주세요. 챗봇을 종료합니다.")
        return

    # PDF에서 6가지 유형에 대한 컨텍스트 추출 (쿼리 생성)
    rag_query = "진단 검사 결과 기반 6가지 역량 유형(균형형, 이해형, 과신형, 실행형, 판단형, 조심형)의 성향적 특징과 설명 요약"
    context = get_context(retriever, rag_query)

    user_prompt = f"""
아래 조건에 맞는 AX 교육 커리큘럼을 작성해 주세요.
- 교육 대상자: {target_audience}
- 핵심 주제: {topics}

[그룹별 인원수 구성 현황]
- 그룹 1 <균형형, 이해형>: 총 {t_balanced + t_understanding}명 (균형형 {t_balanced}명, 이해형 {t_understanding}명)
- 그룹 2 <과신형, 실행형>: 총 {t_overconfident + t_executing}명 (과신형 {t_overconfident}명, 실행형 {t_executing}명)
- 그룹 3 <판단형, 조심형>: 총 {t_judging + t_cautious}명 (판단형 {t_judging}명, 조심형 {t_cautious}명)

[참고: 진단 유형별 관련 데이터 (PDF 기반 검색 내용)]
{context}
"""

    messages = [
        {"role": "system", "content": system_prompt_template},
        {"role": "user", "content": user_prompt}
    ]

    print("\n⏳ 맞춤형 커리큘럼을 고안하는 중입니다...\n")
    print("-" * 70)
    
    _generate_response(messages)
    
    # 2. 후속 대화 루프
    while True:
        try:
            user_input = input("\n💬 강사에게 추가로 요청할 사항이나 수정할 부분이 있나요? (종료: q): ")
            if user_input.lower() in ['q', 'quit']:
                print("\n챗봇을 종료합니다. 수고하셨습니다!")
                break
                
            if not user_input.strip():
                continue
                
            messages.append({"role": "user", "content": user_input})
            print("\n⏳ 답변을 생성 중입니다...\n")
            print("-" * 70)
            
            _generate_response(messages)
                
        except (KeyboardInterrupt, EOFError):
            print("\n\n챗봇을 강제 종료합니다.")
            break

def _generate_response(messages):
    full_response = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content
        print("\n" + "-" * 70)
        
        # 모델 응답을 히스토리에 추가
        messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        print(f"\n[오류 발생] OpenAI API 통신 중 문제가 발생했습니다: {e}")

if __name__ == "__main__":
    chat_app()
