"""
1. 단어 토큰화(Word Tokenization)

토큰의 기준을 단어(word)로 하는 경우를 단어 토큰화라고 한다.
단순히 마침표(.), 컴마(,), 물음표(?), 세미콜론(;), 느낌표(!) 같은 구두점을 제거하는 작업만으로 토큰화가 되진 않는다.
구두점이나 특수문자가 없을 때 해당 단어가 의미를 잃어버리는 경우도 있기 때문에 단순한 기준으로 문장을 단어 토큰화 하는 것은 어렵다.
"""

"""
2. 토큰화의 기준 선택

아포스트로피(')가 있는 단어는 어떻게 분류해야할까
Don't 혹은 Jone's 등의 문장은 어떻게 토큰화할 수 있을까

Don t, Jone s
Do n't, Jones
등등 다양한 방법이 있을 것이다.
직접 토큰화 기준을 설계할 수 있겠지만 기존 도구의 목적과 나의 기준이 같다면 해당 도구를 그냥 사용하는 것이 좋을 것이다.
NLTK는 어떻게 처리하는지 살펴보자.
"""

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

word = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."

print('단어 토큰화1 :', word_tokenize(word))
# 'Do', "n't", 'Jone', "'s"로 분리한다.

print('단어 토큰화2 :', WordPunctTokenizer().tokenize(word))
# 'Don', "'", 't', 'Jone', "'", 's'
# 이 토크나이저는 모든 구두점을 별도로 분류하는 특징을 갖고 있다.

print('단어 토큰화3 :', text_to_word_sequence(word))
# "don't", "jone's"
# 모든 알파벳을 소문자로 바꾸고 구두점을 제거하지만 아포스트로피는 보존을 한다.


"""
3. 토큰화에서 고려해야할 사항

위에서 본 모듈들의 작업들이 단순히 코퍼스에서 구두점을 제외하고 공백 기준으로 잘라내는 작업이라고 간주할 순 없다.

마침표로 예를 들자면 마침표(.)는 문장의 경계를 알 수 있는데 도움이 되기에 제외하지 않을 수도 있으며
$45.55와 같은 가격을 의미하는 단어를 토큰화할 때 45와 55로 분류하고 싶지 않을 수도 있다.

띄어쓰기로 예를 들자면 rock n roll 처럼 하나의 단어이지만 중간에 띄어쓰기가 존재하는 경우도 있다.

위의 예시뿐만 아니라 많은 구두점이 문장에 따라 쓰임새가 달라지고 그 의미하는 바가 달라지기 때문에
토큰화 작업은 섬세한 알고리즘이 필요하다.

다만 표준으로 사용되는 토큰화 방법이 있다.

기준이 완벽하다한들 코퍼스가 오타가 많거나 문장의 구성이 엉망이라면 소용이 없을 수도 있다.
"""

"""
4. 문장 토큰화

토큰의 단위를 문장으로 정하는 경우다.

위와 비슷한 사유로 단순히 마침표(.), 느낌표, 물음표를 기준으로 문장을 잘라내는 것은 좋지 않다.

한국어의 경우 박상길님이 개발한 KSS(Korean Sentence Splitter)를 사용해보자.
"""

from nltk.tokenize import sent_tokenize

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :', sent_tokenize(text))
# 단순히 .로 토큰화를 했다면 Ph.D 같은 부분도 문장으로 처리됬을 것이다.


import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :', kss.split_sentences(text))

"""
5. 한국어 토큰화의 어려움.

한국어는 어절을 단위로 토큰화를 하면 단어 기준으로 토큰화가 되지 않는다.
조사, 어미 등을 붙여서 말을 만드는 교착어기 때문이다.
ex) 그는, 그가, 그를, 그와, 그에게
따라서 토큰화 시 이런 조사 등을 분리해줄 필요가 있다.

또한 한국어 토큰화 시 형태소(morpheme)란 개념을 이해해야한다.
형태소란 뜻을 가진 가장 작은 말의 단위를 의미.

자립 형태소 : 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소. 그 자체로 단어가 된다. 체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사 등이 있다.
의존 형태소 : 다른 형태소와 결합하여 사용되는 형태소. 접사, 어미, 조사, 어간을 말한다.

ex) 에디가 책을 읽었다
자립 형태소 : 에디, 책
의존 형태소 : -가, -을, 읽-, -었, -다

위 처럼 한국어는 어절 보다는 형태소 토큰화를 수행하는 것이 낫다.

한국어 코퍼스는 띄어쓰기가 영어보다 잘 지켜지지 않는다는 점도 토큰화의 어려움에 기여한다.
"""

"""
6. 품사 태깅(Part-of-speech tagging)

'못'은 명사의 의미도 있고 동사 앞에 붙으면 '못 한다', '못 먹는다' 같은 부정의 의미로 사용된다.
이 처럼 똑같은 단어지만 의미가 다를 때 각 단어가 어떤 품사로 쓰였는지 구분하는 작업을 품사태깅이라고 한다.
"""

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :', tokenized_sentence)
print('품사 태깅 :', pos_tag(tokenized_sentence))
# PRP: 인칭 대명사, VBP: 동사, RB: 부사, VBG: 현재부사, IN: 전치사, NNP: 고유 명사, NNS: 복수형 명사, CC: 접속사, DT: 관사

# 한국어 자연어 처리는 KoNLPy 패키지의 형태소 분석기 Okt(Open Korea Text), 메캅(Mecab), 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma)들을 사용 가능하다.

from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

word = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"
print('OKT 형태소 분석 :', okt.morphs(word))
print('OKT 품사 태깅 :', okt.pos(word))
print('OKT 명사 추출 :', okt.nouns(word))

print('꼬꼬마 형태소 분석 :', kkma.morphs(word))
print('꼬꼬마 품사 태깅 :', kkma.pos(word))
print('꼬꼬마 명사 추출 :', kkma.nouns(word))

# konlpy의 형태소 분석기들은 기본적으로 위 3개의 method를 갖고 있다.
# 실행했을 때 결과는 다르게 나오니 사용하고자 하는 용도에 따라 선택하면 된다.
