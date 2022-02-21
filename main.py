from scripts import define_grad_descent_function


w_0 = int(input("Starting w_0: "))
w_1 = int(input("Starting w_1: "))
vector_w = [w_0, w_1]

alpha = float(input("Learning rate: "))

w = define_grad_descent_function.descent(vector_w,vector_w,alpha)
print(w)