"""9. The matrix F is formed as follows: copy A to it and if B to the number of rows consisting of some zeros
in even columns greater than the sum of positive elements in even rows, then swap E and C symmetrically,
otherwise swap B and E asymmetrically. At the same time, the matrix A does not change. After that,
if the determinant of the matrix A is greater than the sum of the diagonal elements of the matrix F,
then the expression is calculated: A-1*AT – K * F-1, otherwise the expression (AT +G-FT)*K is calculated,
where G is the lower triangular matrix obtained from A. Output as A, F and all matrix operations are formed sequentially."""
import numpy as np
import time
import random
from matplotlib import pyplot as plt
import seaborn as sns
N = int(input("enter even a number rows/columns of the square matrix, that greater than 3 and less than 180: "))
while (N < 4) or (N % 2 != 0) or (N > 180):
    N = int(input("enter a number rows/columns of the square matrix, that greater than 3 and less than 180: "))
K = int(input("Enter ratio: "))
start = time.time()
A = np.zeros((N,N), dtype = int)                                        #создаём матрицы А, F и результирующую
F = np.zeros((N,N), dtype = int)
result_matrix = np.zeros((N,N), dtype = int)
for i in range(N):                                                      #заполняем матрицу А
    for j in range(N):
        #A[i][j] = int(input())
        A[i][j] = random.randint(-10,10)
time_next = time.time()
print("matrix A: \n", A, "\ntime: ", time_next-start)
F = A.copy()                                                            #заполняем матрицу F, копируя значения с А
time_prev = time_next
time_next = time.time()
print("matrix F:\n", F, "\ntime: ", time_next-time_prev)
count_zeroes = 0                                                        #счётчик количества строк в области B, состоящих из одних нулей в чётных столбцах
for i in range(0,int(N/2),1):
    sum1 = 0                                                            #счётчик количества строк в области B, состоящих из одних нулей в чётных столбцах(где нули)
    sum2 = 0                                                            #счётчик количества чётных столбцов в области B
    for j in range(1,int(N/2),2):
        if (A[i][j] == 0):
            sum1 += 1
        sum2 += 1
    if sum1 == sum2:
        count_zeroes += 1
sum_numbers = 0                                                         #сумма положительных элементов чётных строк в области B
for i in range(1,int(N/2),2):
    for j in range(0,int(N/2),1):
        if A[i][j] > 0:
            sum_numbers = sum_numbers + A[i][j]
if count_zeroes > sum_numbers:
    print("first condition was fulfilled")                              #первое условие выполнилось, меняются симметрично области C и E
    for i in range(int(N/2)):
        for j in range(int(N/2)):
            F[i][int(N/2)+j] = A[N-i-1][int(N/2)+j]
            F[N-i-1][int(N/2)+j] = A[i][int(N/2)+j]
else:
    print("second condition was fulfilled")                             #второе условие выполнилось, меняются несимметрично области B и E
    for i in range(int(N/2)):
        for j in range(int(N/2)):
            F[i][j] = A[int(N/2)+i][int(N/2)+j]
            F[int(N/2)+i][int(N/2)+j] = A[i][j]
time_prev = time_next
time_next = time.time()
print("New matrix F: \n", F, "\ntime: ", time_next-time_prev)
if (np.linalg.det(A) == 0) or (np.linalg.det(F) == 0):                  #проверяется условие, чтобы определители матриц не были равны нулю, т.к. дальнейшее условие не получится, мы не сможем вичислить обратную матрицу
    print("further calculation is impossible. the determinant is zero")
elif np.linalg.det(A) > sum(np.diagonal(F)):                            #проверяется первое условие, если определитель матрицы А больше суммы диагональных элементов новой матрицы F
    print("the first formula: A^-1*AT – K * F^-1\n")
    result_matrix = np.dot(np.linalg.inv(A), np.transpose(A)) - K * np.linalg.inv(F)
else:
    print("matrix G:\n", np.tril(A))                                    #создаётся матрица G-нижняя треугольная матрица, полученная из А
    print("the second formula: (AT +G-FT)*K\n")
    result_matrix = (np.transpose(A) + np.tril(A) - np.transpose(F)) * K
finish = time.time()
result_time = finish - start
print("result matrix: \n", result_matrix)
print("\nProgram time: " + str(result_time) + " seconds.")


fig1, ax1 = plt.subplots()
plt.title("plot")
plt.xlabel("line number")
plt.ylabel("element value")
fig2, ax2 = plt.subplots()
plt.title("scatter")
fig3, ax3 = plt.subplots()
plt.title("stem")
plt.xlabel("line number")
plt.ylabel("max element in line")
ax1.plot(result_matrix)
ax1.set_xlim(0,N-1)
ax2.scatter(result_matrix, result_matrix)
max_in_row = np.zeros(N)
for i in range(N):
    for j in range(N):
        if result_matrix[i][j] > max_in_row[i]:
            max_in_row[i] = result_matrix[i][j]
ax3.stem(max_in_row)
fig1, ax1 = plt.subplots()
plt.title("scatterplot")
plt.xlabel("line number")
plt.ylabel("element value")
sns.scatterplot(data=result_matrix, ax=ax1)
plt.show()
