# 2주차 정리 노트

상태: 시작 전

# Q1) 단일 퍼셉트론으로 이문제를 풀 수 있을까?

# AND 게이트

---

동현)

그런데 0.7 임계점을 무시하고 저런 값이 나올 수 있으니까 질문이 들어온거 아닌가?

(0, 0), (0, 1), (1, 0) 점과 (1, 1)을 나누는 기울기가 -1인 그래프를 구하고 그걸 이용해서 가중치를 구할 수 있지 않나?

⇒ 그러면 y = -x +1 과 y = -x + 2 사이에 기울기가 -1인 직선 그래프를 구하고 그걸 이용해 가중치를 구하면 되지 않나?

⇒ 근데 그러면 예를 들어서 y = -x + 1.5라고 하면 이 선위에 존재하는 한 점을 가중치로 두고 (1, 0.5) 조건에 대입해 보면 0 , 0 일때는 1, 1일때는 1이 맞는데 **0, 1과 1, 0문제가 됨**

⇒ 결국 b 가 필수 적으로 들어가야지 계산이 될듯

<aside>
✔️ **간단하게 생각해 보면 증명도 가능함**

각 가중치를 a와 b라고 한다면 a≤0, b≤0, a+b>0이고 

a와 b는 0 또는 음수이지만 a+b > 0을 만족시키려면 a와 b의 합이 양수여야 한다

따라서 이걸 만족하는 a, b 값은 없다

</aside>

**w1을 1, w2를 0.5하고 하면 b는 -1보다 작거나 같아야함** 

![Untitled](2%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5%20%E1%84%82%E1%85%A9%E1%84%90%E1%85%B3%208809dd7694b1450b88c570f47fd6ceb8/Untitled.png)

유성) 

⇒ w1,w2 중에서 무조건 하나는 마이너스 값이 들어가야된다고 생각.

 그런데 그렇다면 01,10 부분에서 값이 맞지 않는다.

![https://blog.kakaocdn.net/dn/bkHXbu/btrmSa8KO8e/BACdm9lg9TJBzESTRNSjf0/img.png](https://blog.kakaocdn.net/dn/bkHXbu/btrmSa8KO8e/BACdm9lg9TJBzESTRNSjf0/img.png)

![https://blog.kakaocdn.net/dn/bY0SDa/btrmSxC4jZy/39ueAEaiY8MLxEK6kwB7y0/img.png](https://blog.kakaocdn.net/dn/bY0SDa/btrmSxC4jZy/39ueAEaiY8MLxEK6kwB7y0/img.png)

⇒ 임계점을 0.7로 하면 0.5를 넣었을 때, 값이 일치됨. 

그러나 현 문제에서는 0이 중심이기 때문에 다른 방향을 생각해야함.

문제를 푸는 방법은 0.1의 소수점을 처음부터 넣는 하드 한 방식과

프로그래밍으로 인한 소프트한 방법 두 가지를 적절히 섞어서 찾으면 될 듯.

그러나 프로그래밍 없이 하는 방법은 하드 한 방식을 추구한다.

# Q2) 머신러닝을 왜 사용해야 할까?

---

<aside>
❓ **Q&A**

- 동현 - 사람의 능력으로는 할 수 있는 한계가 명확하기에 데이터가 넘쳐흐르는 시대에서 머신 러닝을 사용한 자동화된 시스템은 필수 적이라고 생각함.
    
    ✔️ 유성 →이에 공감한다.
    
- 많은 데이터를 조작하고, 프로그래밍을 할 때 유리할 것 같다

<유성>

- 머신 러닝을 사용하면 인간의 한계를 벗어나 더 다양하고, 빠른 속도로 원하는 데이터를 조작하고, 분석할 수 있다고 생각한다.
- 데이터를 분석하는 데에 머신 러닝이 최적화 되어 있다고 생각한다.
</aside>