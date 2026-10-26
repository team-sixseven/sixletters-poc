# Transformer 계열

- ECG Transformer는 `(3,2000)` 원시 파형 window를 patch token으로 바꿔 attention을 적용한 뒤 8개 window logit을 환자 단위로 평균합니다.
- CNN+Transformer는 각 window를 CNN 임베딩으로 바꾼 뒤 기록 전체에서 균등 추출된 8개 시간 순서 token에 Transformer를 적용합니다.
- 두 구현은 `../models.py`에 있으며 설정은 [`transformer.yaml`](transformer.yaml)에서 읽습니다. 현재 프로젝트용 비교 구조이며 외부 논문의 공식 구현체와 동일하다고 보장하지 않습니다.
