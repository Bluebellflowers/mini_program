n = 1
while True:
    total=[]
    try:
        data = map(int, input().split())
        if n:
            num = int(data)
            n -= 1
        else:
            a, p, q = data
            people = [a, p, q, abs(p - q)]
        total.append(people)
    except:
        break
print(total)