if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

# 로젠브록 함수 정의
def rosenbrock(x0, x1):
    return (1 - x0) ** 2 + 100 * (x1 - x0 ** 2) ** 2

# 초기값 설정
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(-1.0))

# 학습률과 반복 횟수 설정
learning_rate = 0.05
iterations = 10000

# 경사하강법 수행
for i in range(iterations):
    y = rosenbrock(x0, x1)  # 함수 값 계산
    x0.cleargrad()  # 기울기 초기화
    x1.cleargrad()
    y.backward()  # 역전파로 기울기 계산

    # 변수 업데이트
    x0.data -= learning_rate * x0.grad
    x1.data -= learning_rate * x1.grad

    # 100번마다 출력
    if i % 100 == 0:
        print(f"Step {i}: x0 = {x0.data}, x1 = {x1.data}, y = {y.data}")

# 최종 결과 출력
print("최솟값 지점:")
print(f"x0 = {x0.data}, x1 = {x1.data}")

#결과 --- approximate sin ---
0.7070959900908971 , # variable(-0.7071660809820823)