# WAGYU AI

WAGYU AI는 WAGYU 서비스에서 사용할 AI 모델을 학습하고 실험하기 위한 레포지토리입니다.  
한국어 질의응답 데이터셋을 기반으로 LLM을 파인튜닝하며, QLoRA 기반 학습과 데이터 증강 실험을 포함합니다.

> 실제 API 서버는 `WAGYU_aiback` 레포지토리를 사용합니다.

## 주요 기능

- WAGYU 서비스용 AI 모델 파인튜닝
- QLoRA 기반 저비용 LLM 학습
- Qwen2.5-7B-Instruct 기반 SFT 학습
- 한국어 질의응답 데이터셋 로드 및 학습 포맷 변환
- 형태소 분석 기반 데이터 증강 실험
- 학습된 모델 및 토크나이저 저장

## 기술 스택

- Python
- PyTorch
- Transformers
- Datasets
- TRL
- PEFT
- BitsAndBytes
- KoNLPy
- uv

## 프로젝트 구조

```bash
WAGYU_AI/
├── main.py                 # QLoRA 기반 모델 학습 코드
├── app.py                  # 실험용 FastAPI 추론 코드
├── requirements.txt        # Python 패키지 목록
├── uv.lock                 # uv 의존성 잠금 파일
├── augment/
│   └── 통합_데이터셋_증강.json   # 학습용 증강 데이터셋
└── README.md
```

## 설치 및 실행 준비

### 1. 저장소 클론

```bash
git clone https://github.com/capstone-WAGYU/WAGYU_AI.git
cd WAGYU_AI
```

### 2. 가상환경 생성

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

또는 `uv`를 사용하는 경우:

```bash
pip install uv
uv sync
```

## Java 설치 필요

KoNLPy 기반 형태소 분석기를 사용하는 경우 Java JRE 또는 JVM이 필요합니다.

Java가 설치되어 있지 않으면 형태소 분석 과정에서 JVM 관련 오류가 발생할 수 있습니다.

## 데이터셋 준비

학습 코드는 아래 경로의 JSON 데이터셋을 사용합니다.

```bash
augment/통합_데이터셋_증강.json
```

데이터셋은 Git에 포함하지 않는 것을 권장합니다.

예상 데이터 형식은 다음과 같습니다.

```json
[
  {
    "question": "사용자 질문",
    "answer": "모델이 학습할 답변"
  }
]
```

## 모델 학습

```bash
python main.py
```

`main.py`는 `Qwen/Qwen2.5-7B-Instruct` 모델을 기반으로 QLoRA 학습을 수행합니다.

학습 결과는 기본적으로 아래 경로에 저장됩니다.

```bash
qwen2.5-7b-instruct/
```

## 학습 설정

현재 학습 코드는 다음 설정을 사용합니다.

| 항목 | 값 |
|---|---|
| Base Model | `Qwen/Qwen2.5-7B-Instruct` |
| 학습 방식 | QLoRA |
| Quantization | 4bit NF4 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Max sequence length | 2048 |
| Epoch | 3 |
| Learning rate | 2e-4 |
| Max steps | 2000 |

## 학습 데이터 포맷 변환

학습 전 각 데이터는 Chat Template 형식으로 변환됩니다.

```python
messages = [
    {"role": "system", "content": "너는 WAGYU 서비스를 보조하는 AI 어시스턴트다."},
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer},
]
```

이후 tokenizer의 `apply_chat_template`을 사용해 학습용 텍스트로 변환합니다.

## 실험용 API 실행

`app.py`에는 학습된 모델을 불러와 FastAPI로 추론하는 실험 코드가 포함되어 있습니다.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Health Check

```http
GET /
```

### 질문 요청

```http
POST /ask
```

#### Request Body

```json
{
  "question": "강아지가 밥을 잘 안 먹어요. 어떻게 해야 하나요?",
  "max_newtokens": 256
}
```

## 주의사항

현재 `app.py`는 실험용 코드입니다.  
실제 서비스 API 서버는 `WAGYU_aiback` 레포지토리를 기준으로 사용해야 합니다.

또한 `app.py` 내부 모델 경로가 로컬 절대 경로로 작성되어 있으므로, 실행 환경에 맞게 수정해야 합니다.

```python
MODEL = "/home/chldlsrb08/final_model"
```

## Git 관리 참고

다음 파일 및 디렉토리는 Git에 포함하지 않는 것을 권장합니다.

```bash
.venv/
__pycache__/
augment/통합_데이터셋_증강.json
```

## Repository 역할

| Repository | 역할 |
|---|---|
| WAGYU_AI | 모델 학습, 데이터 증강, AI 실험 |
| WAGYU_aiback | 실제 AI API 서버 |

## License

별도 라이선스가 명시되어 있지 않습니다.
