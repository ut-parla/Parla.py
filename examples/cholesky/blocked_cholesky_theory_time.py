

#Experimentally gathered kernel times (default: 2000 x 2000 matrices)
potrf = 0.02
gemm = 0.047
syrk =  gemm
trsm = 0.03

crit_time = 0
easy_time = 0

#How many GPUs
p = 4.0

#How many blocks in a side
n = 14

levels = n

data_count = 0
for i in range(1, levels):

    easy_time += potrf
    data_count += (n-i) * ((p-1)/p)
    easy_time += (n-i)/p * trsm
    easy_time += (n-i)/p * syrk
    easy_time += (0.5 * (n - 1 - i)*(n-i) )/p * gemm
    data_count += (0.5 * (n-1-i)*(n-i)) * ((p-1)/p)

    crit_time += potrf
    crit_time += (n-i)/p * trsm

    start = (n - i) % p
    current = 0
    j = 0
    #print("-------", start)
    while current < (n - i):
        current = start + j*p
        print(current)
        crit_time += gemm * (current - 1)  + syrk * (1)
        j = j + 1

print("Maximal Parallelism Time: ", easy_time)
print("Critical Path Time: ", crit_time)
print("Communication Time: ", (data_count * 2000 * 2000 * 8)/(6*1e9))


