# import tensorflow as tf
"""
텐서플로우는 구글이 2015년에 공개한 머신 러닝 오픈소스 라이브러리.
머신 러닝과 딥 러닝을 직관적이고 손쉽게 할 수 있도록 설계됨.
"""
# import keras
"""
케라스(Keras)는 딥 러닝 프레임워크인 텐서플로우에 대한 추상화 된 API를 제공.
텐서플로우 코드를 훨씬 간단하게 작성할 수 있게 해줌.
tensorflow 내부에도 있어서 tf.keras를 쓰는 것이 권장 됨.
"""
# import gensim
"""
젠심은 머신 러닝을 사용하여 토픽 모델링과 자연어 처리 등을 수행할 수 있게 해주는 오픈 소스 라이브러리.
토픽 모델링이란?
문서 집합의 추상적인 "주제"를 발견하기 위한 통계적 모델 중 하나.
텍스트 본문의 숨겨진 의미구조를 발견하기 위해 사용되는 텍스트 마이닝 기법 중 하나.
"""
# import sklearn
"""
사이킷런은 python의 머신러닝 라이브러리. 사이킷런을 통해 나이브 베이즈 분류, 서포트 벡터 머신 등 다양한 머신 러닝 모듈 호출 가능.
또한 사이킷런에는 머신러닝을 연습하기 위한 아이리스 데이터, 당뇨병 데이터 등 자체 데이터 또한 제공.

서포트 벡터 머신이란?
기계 학습의 분야 중 하나로 패턴 인식, 자료 분석을 위한 지도 학습 모델.

나이브 베이즈 분류란?
특성들 사이의 독립을 가정하는 베이즈 정리를 적용한 확률 분류기의 일종.
(베이즈 정리) https://ko.wikipedia.org/wiki/%EB%B2%A0%EC%9D%B4%EC%A6%88_%EC%A0%95%EB%A6%AC
"""
# import nltk
# nltk.download()
"""
자연어 처리를 위한 파이썬 패키지
NLTK의 기능을 제대로 사용하기 위해서는 NLTK Data를 추가적으로 설치 필요.
nltk.download() 코드로 설치.
"""
# import konlpy
"""
코엔엘파이(KoNLPy)는 한국어 자연어 처리를 위한 형태소 분석기 패키지.
JAVA로 구성되어있어 JDK 1.7 이상의 버전과 JPype 설치 필요.
(JAVA 설치 주소) https://www.oracle.com/technetwork/java/javase/downloads/index.html
0.6.0 설치 시점에는 JPype 자동으로 설치 됨.
"""
import pandas as pd
"""
(pandas docs) http://pandas.pydata.org/pandas-docs/stable/

pandas는 세 가지의 데이터 구조를 사용.
1. 시리즈(Series)
2. 데이터프레임(DataFrame)
3. 패널(Panel)
"""

# series
sr = pd.Series([17000, 18000, 1000, 5000], index=["피자", "치킨", "콜라", "맥주"])
# 1차원 배열의 값(values)에 각 값에 대응되는 인덱스(index)를 부여할 수 있는 구조
print(f'시리즈 전체 : {sr}')
print(f'시리즈의 값 : {sr.values}')
print(f'시리즈의 인덱스 : {sr.index}')

# dataframe
# 데이터프레임은 2차원 리스트를 매개변수로 전달합니다. 2차원이므로 행방향 인덱스(index)와 열방향 인덱스(column)가 존재. 즉 행과 열을 가지는 자료구조.
# 데이터프레임은 열(columns)까지 추가되어 열(columns), 인덱스(index), 값(values)으로 구성

values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)
print(df)
print(f'데이터프레임의 인덱스 : {df.index}')
print(f'데이터프레임의 열이름: {df.columns}')
print('데이터프레임의 값 :')
print(df.values)

# 데이터프레임은 리스트(List), 시리즈(Series), 딕셔너리(dict), Numpy의 ndarrays와
# 외부 파일인 csv 파일, 엑셀 파일, 심지어 html 등 여러 방법으로 생성 가능하다.

import numpy as np
"""
넘파이는 수치 데이터를 다루는 파이썬 패키지.
다차원 행렬 자료구조인 ndarray를 통해 벡터 및 행렬을 사용하는 선형 대수 계산에서 주로 사용.
C로 구현. 속도면에서 순수 파이썬에 비해 압도적으로 빠름.
(pandas 설치 시 같이 설치 됨.)
"""

vec = np.array([1, 2, 3, 4])    # 1차원 배열
mat = np.array([[10, 20, 30], [60, 70, 80]])   # 2차원 배열
# np.array()는 리스트, 튜플, 배열로 부터 ndarray를 생성.

print(f'vec의 타입 : {type(vec)}')
print(f'mat의 타입 : {type(mat)}')
# 타입은 동일하게 ndarray.

print(vec.ndim)
# 축의 개수 출력
print(vec.shape)
# 크기 출력

print(mat.ndim)
# 축의 개수 출력
print(mat.shape)
# 크기 출력

# Numpy 배열에는 축의 개수(ndim)와 크기(shape)라는 개념이 존재하는데, 배열의 크기를 정확히 숙지하는 것은 딥 러닝에서 매우 중요.

zero_mat = np.zeros((2, 3))
print(zero_mat)
# 모든 값이 0인 2x3 배열 생성.

one_mat = np.ones((2, 3))
print(one_mat)
# 모든 값이 1인 2x3 배열 생성.

same_value_mat = np.full((2, 2), 7)
print(same_value_mat)
# 모든 값이 특정 상수인 배열 생성. 이 경우 7.

eye_mat = np.eye(3)
print(eye_mat)
# 대각선 값이 1이고 나머지 값이 0인 2차원 배열을 생성.

random_mat = np.random.random((2, 2))
print(random_mat)
# 임의의 값으로 채워진 배열 생성

range_vec = np.arange(10)
print(range_vec)
# np.arange(n)은 0부터 n-1까지의 값을 가지는 배열을 생성.

range_n_step_vec = np.arange(1, 10, 2)
print(range_n_step_vec)
# np.arange(i, j, k)는 i부터 j-1까지 k씩 증가하는 배열을 생성.

reshape_mat = np.array(np.arange(30)).reshape((5, 6))
print(reshape_mat)
# np.reshape()은 내부 데이터는 변경하지 않으면서 배열의 구조를 변경.

# 슬라이싱
mat = np.array([[1, 2, 3], [4, 5, 6]])

slicing_mat = mat[0, :]
print(slicing_mat)
# 첫번째 행 출력
slicing_mat = mat[:, 1]
print(slicing_mat)
# 두번째 행 출력

mat = np.array([[1, 2], [4, 5], [7, 8]])
indexing_mat = mat[[2, 1], [0, 1]]
# mat[[2행, 1행], [0열, 1열]]
# 각 행과 열의 쌍을 매칭 하면 2행 0열, 1행 1열의 두 개의 원소.
# [7 5] 가 출력 됨.

# 연산
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
result = x + y
print(result)
# [5 7 9] 같은 열의 값 끼리 계산.
# result = np.add(x, y)와 동일.

result = x - y
print(result)
# [-3 -3 -3]
# result = np.subtract(x, y)와 동일.

result = result * x
print(result)
# [-3 -6 -9]
# result = np.multiply(result, x)와 동일.

result = result / x
print(result)
# [-3. -3. -3.]
# result = np.divide(result, x)와 동일.

mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
mat3 = np.dot(mat1, mat2)
print(mat3)
# [[19 22] [43 50]]
# Numpy에서 벡터와 행렬곱 또는 행렬곱을 위해서는 dot()을 사용

import matplotlib.pyplot as plt
"""
맷플롯립은 데이터를 차트(chart)나 플롯(plot)으로 시각화하는 패키지.
데이터 분석 이전에 데이터 이해를 위한 시각화나, 데이터 분석 후에 결과를 시각화하기 위해서 사용.
"""

plt.title('test')
# 그래프 제목
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
# 데이터 (x축, y축)
plt.plot([1.5, 2.5, 3.5, 4.5], [3, 5, 8, 10])
# plot 명령어 사용 시 마다 라인이 추가 됨.
plt.xlabel('hours')
# x축 라벨
plt.ylabel('score')
# y축 라벨
plt.legend(['A student', 'B student'])
# 범례
plt.show()
# 그래프 렌더링

