# 🙌목적

- Playground Series - Season 3, Episode 4 : 주어진 데이터 세트를 이용해 신용카드 사기 감지 예측
- Playground Series - Season 3, Episode 2 : 주어진 데이터 세트를 이용해 뇌졸중 예측

# ❓데이터 수집
- Kaggle 경쟁대회 Playground 데이터
- Kaggle에서 추가로 제공한 Original 데이터

# 😀참여인원
- [두마리토끼조]
- 지승엽 Episode 4, 2
- 고종현 Episode 2, 6
- 박상욱 Episode 5, 6
- 이건우 Episode 5
- 김예담 Episode 6

# 💳Episode 4

## 데이터 분석 결과
### 타겟값 분포
![정답값 분포](https://user-images.githubusercontent.com/125621591/229060796-2fb21c7a-0928-4998-bd2f-75de5ae38d0b.png)
### 피처별 train, test, orig_train 분포도
![타겟기준 피처분포](https://user-images.githubusercontent.com/125621591/229063491-f727980c-a093-4609-9cb5-ad7e3c079b66.png)
- train, test, orig_train의 분포를 비교했더니 'Time'만 분포가 매우 달랐음
  - 'Time' 컬럼만 제거 후 orig_train 사용

## 베이스라인 모델 구축 및 기본학습
- 학습 시키기 전, 스케일링 함수와 점수처리 함수 생성

### 모델 학습
1. LogisticRegression
2. XGB
3. LGBM
- LGBM이 성능이 가장 좋았고, 학습시간도 빨랐음

### 스케일링
1. 스케일링 하지 않음
2. StandardScaler
3. MinMaxScaler
- MinMaxScale이 가장 성능이 좋았음

## 모델 성능 향상
- 'Time' 피처 제거
- 하이퍼파라미터 튜닝
  - Optuna 이용

### 최적 모델 선정
- LGBM과 MinMaxScaler를 이용

## 결과
- 최종적으로 약 83%의 정확도를 얻을 수 있었음
<img src="https://user-images.githubusercontent.com/125621591/229071188-2090ff3e-28d9-42a1-a426-d69623709f1d.png" width="500" height="200"/>

# 🧠Episode 2

## 데이터 분석 결과
### 이진형 분포도
![이진형분포도](https://user-images.githubusercontent.com/125621591/229090941-a376dfb4-653e-4203-b553-a6d9f83a3d44.png)
- Residence_type의 Urban과 Rural의 정답 비율이 같음
  - Residence_type 피처 제거
  

### 순서형 분포도
<img src="https://user-images.githubusercontent.com/125621591/229092292-56011055-2867-4a61-a84b-fe57cfef590d.png" width="500" height="400"/>
- never somked -> unknown -> formerly smoked -> smokes로 순서 부여

## 베이스라인 모델 구축 및 기본학습
- 결측치 대체, 피처값 대체, 순서 부여, 인코딩 수행

### 인코딩
1. 라벨 인코딩
2. 원-핫 인코딩
- 원-핫 인코딩이 성능이 좋았음

### 스케일링
1. 스케일링 하지 않음
2. MinMaxScaler
- MinMaxScaler가 성능이 좋았음

## 모델 성능 향상
- 모델 : XGB, LGBM, CATBOOST
- 결측치 대체, 피처값 대체, 순서 부여
- 원-핫 인코딩
- MinMaxScaler

### 최적 모델 선정
- PYCARET을 이용해 자동으로 모델 선정, 파라미터 찾기
  -> XGB로 선정

## 결과
- 최종적으로 약 90%의 정확도를 얻을 수 있었음
<img src="https://user-images.githubusercontent.com/125621591/229094413-772f8963-be67-4caf-8307-a3ebb98dcbb4.png" width="500" height="200"/>
