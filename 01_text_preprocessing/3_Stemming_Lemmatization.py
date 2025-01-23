"""
정규화 기법 중 코퍼스의 단어 개수를 줄이는 기법에는 표제어 추출과 어간 추출이 있다.

단어는 다르나 의미가 같은 것들을 하나의 단어로 일반화 시키는 작업.
해당 방법은 단어의 빈도수를 기반으로 문제를 풀고자 하는 BoW(Bag of Words)
표현을 사용하는 자연어 처리에 문제에서 주로 사용된다.

전처리 중 정규화의 지향점은 언제나 코퍼스의 복잡성을 줄이는 것이다.

1. 표제어 추출(Lemmatization)

'표제어'는 '기본 사전형 단어' 정도의 의미를 갖는다.
am, are, is는 스펠링이 다르지만 뿌리 단어는 be라고 볼 수 있다.
이러한 뿌리 단어를 찾아가서 단어의 개수를 줄일 수 있는지 판단하는 것이다.

표제어 추출을 하는 가장 섬세한 방법은 단어의 형태학적 파싱을 먼저 진행하는 것이다.
형태소 : 의미를 가진 가장 작은 단위
형태학(morphology): 형태소로부터 단어들을 만들어가는 학문
형태소의 종류로 어간(stem)과 접사(affix)가 존재한다.

어간 : 단어의 의미를 담고 있는 단어의 핵심 부분
접사 : 단어에 추가적인 의미를 주는 부분

즉 형태학적 파싱은 이 두 가지 구성 요소를 분리하는 작업을 말한다.
(cats에서 cat은 어간 s는 접사)

NLTK에서는 표제어 추출을 위한 도구인 WordNetLemmatizer를 지원.
"""

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print('표제어 추출 후 :', [lemmatizer.lemmatize(word) for word in words])
# has -> ha, dies -> dy 같이 의미를 알 수 없는 단어를 출력하기도 한다.

print(lemmatizer.lemmatize('dies', 'v'))
print(lemmatizer.lemmatize('watched', 'v'))
print(lemmatizer.lemmatize('has', 'v'))
# 위 처럼 단어가 동사 품사라는 사실을 알려주면 정확한 표제어를 추출한다.


"""
2. 어간 추출(Stemming) 

어간을 추출하는 작업. 
형태학적 분석을 단순화한 버전 혹은 정해진 규칙만 보고 단어의 어미를 자르는 어림짐작의 작업이라고 볼 수도 있다.
섬세한 작업이 아니기에 어간 추출 후 결과 단어는 사전에 존재하지 않는 단어일 수도 있다.
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
# 어간 추출 알고리즘 중 하나인 포터 알고리즘(Porter Algorithm)

sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
tokenized_sentence = word_tokenize(sentence)

print('어간 추출 전 :', tokenized_sentence)
print('어간 추출 후 :', [stemmer.stem(word) for word in tokenized_sentence])
# 추출 결과에는 사전에 없는 단어들도 많이 포함되어 있다.

"""
ALIZE → AL
ANCE → 제거
ICAL → IC
여러 규칙 중 일부는 위 같은 규칙을 가진다.

Porter 알고리즘의 상세 규칙은 마틴 포터의 홈페이지에서 확인할 수 있다.
https://tartarus.org/martin/PorterStemmer/index.html

어간 추출 속도는 표제어 추출보다 일반적으로 빠르며 포터 어간 추출기는 정밀하게 설계되어
정확도가 높으므로 영어 자연어 처리에서 어간 추출을 하고자 한다면 가장 준수한 선택이다.
"""

words = ['formalize', 'allowance', 'electricical']

print('어간 추출 전 :', words)
print('어간 추출 후 :', [stemmer.stem(word) for word in words])
# 'formal', 'allow', 'electric'이 출력 됨.

from nltk.stem import LancasterStemmer
"""
NLTK는 랭커스터 스태머(Lancaster Stemmer) 알고리즘 또한 지원한다.
알고리즘 별로 어간 추출의 결과는 전혀 다르기에 내가 가진 코퍼스에 더 적합한 알고리즘을 판단할 수 있어야한다.

다만 모든 알고리즘이 완벽할순 없다.
organization을 어간 추출했을 때 완전히 다른 단어인 organ이 추출되며 organ 역시 어간 추출 시 organ이 된다.
이는 의미가 동일한 경우에만 같은 단어를 얻기를 원하는 정규화의 목적에는 맞지 않는다.
"""

"""
3. 한국어에서의 어간 추출

한국어는 아래와 같이 5언 9품사의 구조를 가진다.

언	품사
체언	명사, 대명사, 수사
수식언	관형사, 부사
관계언	조사
독립언	감탄사
용언	동사, 형용사

용언에 해당하는 '동사'와 '형용사'는 어간(stem)과 어미(ending)의 결합으로 구성된다.

(1) 활용(conjugation)
용언의 어간이 어미를 가지는 일을 말한다.

어간(stem) : 용언(동사, 형용사)을 활용할 때, 원칙적으로 모양이 변하지 않는 부분. 활용에서 어미에 선행하는 부분.
때론 어간의 모양도 바뀔 수 있다.(예: 긋다, 긋고, 그어서, 그어라).
어미(ending) : 용언의 어간 뒤에 붙어서 활용하면서 변하는 부분이며, 여러 문법적 기능을 수행.

활용은 규칙 활용, 불규칙 활용으로 나뉜다.

(2) 규칙 활용
규칙 활용은 어간이 어미를 취할 때, 어간의 모습이 일정하다.
ex) 잡/어간 + 다/어미
위 경우 규칙 기반으로 어미를 단순히 분리하면 어간 추출이 된다.

(3) 불규칙 활용
불규칙 활용은 어간이 어미를 취할 때 어간의 모습이 바뀌거나 취하는 어미가 특수한 어미일 경우다.

예를 들어
‘듣-, 돕-, 곱-, 잇-, 오르-, 노랗-’ 등이 ‘듣/들-, 돕/도우-, 곱/고우-, 잇/이-, 올/올-, 노랗/노라-’
와 같이 어간의 형식이 달라지는 일이 있거나
‘오르+ 아/어→올라, 하+아/어→하여, 이르+아/어→이르러, 푸르+아/어→푸르러’
와 같이 일반적인 어미가 아닌 특수한 어미를 취하는 경우 불규칙활용을 하는 예에 속한다.

위 경우 단순한 분리만으로 어간 추출이 되지 않고 좀 더 복잡한 규칙을 필요로 한다.
https://namu.wiki/w/한국어/불규칙%20활용 (더 많은 불규칙 활용의 예)
"""