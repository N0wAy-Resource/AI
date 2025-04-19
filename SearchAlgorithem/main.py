import random
import math

# 定义目标函数
def target_function(x, y, a, b, c, d, e, f):
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f

# 模拟退火算法
def simulated_annealing(a, b, c, d, e, f, initial_temp=1000, cooling_rate=0.9, num_iterations=10):
    # 初始解
    x, y = random.uniform(-10, 10), random.uniform(-10, 10)
    current_cost = target_function(x, y, a, b, c, d, e, f)

    # 最优解
    best_x, best_y = x, y
    best_cost = current_cost

    # 模拟退火过程
    for _ in range(num_iterations):
        # 生成新解
        new_x = x + random.uniform(-1, 1)
        new_y = y + random.uniform(-1, 1)
        new_cost = target_function(new_x, new_y, a, b, c, d, e, f)

        # 计算代价差
        cost_diff = new_cost - current_cost

        # 如果新解更好，或者以一定概率接受更差的解
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / initial_temp):
            x, y = new_x, new_y
            current_cost = new_cost

            # 更新最优解
            if current_cost < best_cost:
                best_x, best_y = x, y
                best_cost = current_cost

        # 降温
        initial_temp *= cooling_rate

    return best_x, best_y, best_cost


# 随机生成系数，只保留两位小数
a = round(random.uniform(1, 10), 2)
b = round(random.uniform(-10, 10), 2)
c = round(random.uniform(-10, 10), 2)
d = round(random.uniform(-10, 10), 2)
e = round(random.uniform(-10, 10), 2)
f = round(random.uniform(-10, 10), 2)

# 设置迭代次数
num_iterations = 100 # 你可以根据需要调整这个值

# 执行模拟退火算法
best_x, best_y, best_cost = simulated_annealing(a, b, c, d, e, f, num_iterations=num_iterations)
print(f"{a}x^2+{b}y^2+{c}xy+{d}x+{e}y+{f}")
print(f"最优解: x = {best_x}, y = {best_y}")
print(f"最小值: {best_cost}")
