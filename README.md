# 7주차 팀 실습: RAG 챗봇 만들기

2024 KB부동산 보고서 PDF를 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇 프로젝트입니다.

## 프로젝트 개요

- 문서 소스: data/2024*KB*부동산*보고서*최종.pdf
- 앱 프레임워크: Streamlit
- 벡터 저장소: FAISS
- LLM/임베딩: OpenAI (gpt-4o-mini, OpenAIEmbeddings)

사용자가 질문하면 PDF에서 관련 문맥을 검색한 뒤, 검색 결과를 바탕으로 답변을 생성합니다.

## 폴더 구조

```text
rag_chatbot/
├─ app.py
├─ requirements.txt
├─ .env
└─ data/
	├─ 2024_KB_부동산_보고서_최종.pdf
	└─ 서울시_부동산_실거래가_정보.csv
```

## 실행 방법

1. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate
```

2. 패키지 설치

```bash
pip install -r requirements.txt
```

3. 환경 변수 설정

프로젝트 루트의 .env 파일에 아래 내용을 추가합니다.

```env
OPENAI_API_KEY=your_openai_api_key
```

4. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 안내되는 로컬 주소로 접속하면 챗봇을 사용할 수 있습니다.

## 동작 방식

1. PDF 문서를 로드합니다.
2. 문서를 청크로 분할합니다. (chunk_size=1000, chunk_overlap=200)
3. 임베딩을 생성해 FAISS 벡터 DB에 저장합니다.
4. 질문 시 관련 청크를 검색합니다. (k=3)
5. 검색 문맥 + 대화 이력을 바탕으로 답변을 생성합니다.

## 참고 사항

- 첫 실행 시 벡터 인덱스 생성에 시간이 걸릴 수 있습니다.
- 생성된 인덱스는 faiss_db 폴더에 캐시됩니다.
- pypdf가 설치되지 않으면 PDF 로더에서 오류가 발생할 수 있습니다.

## 실습 목표 예시

- 특정 지역/유형의 시장 동향 요약 질문
- 보고서 근거 기반 질의응답 정확도 비교
- 프롬프트/검색 파라미터(k 값 등) 변경 실험
