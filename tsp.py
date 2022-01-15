import numpy as np
import os
import matplotlib.pyplot as plt

# 或者点坐标数据
def get_coordinates(file_path: str) -> np.ndarray:
    # coordinates = None
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
    #     coordinates = np.zeros((len(lines), 2))
    #     for i, line in enumerate(lines):
    #         line = line.strip()
    #         line = line.replace('  ', ' ')
    #         items = line.split()
    #         if len(items) >= 3:
    #             coordinates[i][0] = int(items[1])
    #             coordinates[i][1] = int(items[2])


    coordinates = np.array([[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],
                            [880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],
                            [1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],
                            [725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],
                            [300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],
                            [1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],
                            [420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],
                            [685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],
                            [475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],
                            [830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],
                            [1340.0,725.0],[1740.0,245.0]])

    return coordinates

# 计算点与点之间距离矩阵
def getdismat(coordinates: np.ndarray) -> np.ndarray:
    num = coordinates.shape[0]
    distmat = np.zeros((num, num))
    for i in range(num):
        place_i = coordinates[i]
        for j in range(num):
            place_j = coordinates[j]
            distmat[i][j] = np.sqrt(np.power(place_i[0] - place_j[0], 2) + np.power(place_i[1] - place_j[1], 2))
    return distmat

# 进行模拟退火过程
def sa_run():
    coordinates = get_coordinates('./a280.tsp')
    num = coordinates.shape[0]
    dist_mat = getdismat(coordinates)
    solution_new = np.arange(num)
    solution_current = solution_new.copy()
    solution_best = solution_new.copy()
    value_current = 9999999
    value_best = 9999999
    alpha = 0.99
    t_range = (1, 100)
    markovlen = 1000
    t = t_range[1]

    epochcount = 6
    epoch_best = []
    epoch_current = []
    for epoch in range(epochcount):
        result_best = [] # 记录迭代过程中的最优解
        result_current = []
        while t > t_range[0]:
            for i in range(markovlen):
                # 使用将两个左边逆序的方式产生新解
                while True:
                    loc1 = int(np.ceil(np.random.rand() * (num - 1)))
                    loc2 = int(np.ceil(np.random.rand() * (num - 1)))
                    if loc1 != loc2:
                        break
                solution_new[loc1], solution_new[loc2] = solution_new[loc2], solution_new[loc1]

                value_new = 0
                for j in range(num - 1):
                    value_new += dist_mat[solution_new[j]][solution_new[j + 1]]
                value_new += dist_mat[solution_new[0]][solution_new[num - 1]]
                if value_new < value_current:
                    # 接受该解
                    # print('accept1')
                    value_current = value_new
                    solution_current = solution_new.copy()

                    if value_new < value_best:
                        value_best = value_new
                        solution_best = solution_new.copy()
                else:
                    # 以一定概率接受该解
                    if np.random.rand() < np.exp(-(value_new - value_current) / t):
                        # print('accept2')
                        value_current = value_new
                        solution_current = solution_new.copy()
                    else:
                        # print('not accept')
                        solution_new = solution_current.copy()

                result_current.append(value_current)

            t = alpha * t
            result_best.append(value_best)
            print(f't: {t}, value: {value_current}')
            
        epoch_current.append(result_current)
        epoch_best.append(result_best)

        print(f'epoch {epoch} finish!!!!!!!!')
        solution_new = np.arange(num)
        solution_current = solution_new.copy()
        solution_best = solution_new.copy()
        value_current = 9999999
        value_best = 9999999
        t = t_range[1]

    # print(f'best values: {value_best}')
    # print(f'best solution: {solution_best}')

    fig_col = 3  # 每行多少个子图
    fig_row = epochcount // fig_col
    fig = plt.figure()
    ax = fig.subplots(fig_row, fig_col)
    for r in range(fig_row):
        for c in range(fig_col):
            index = r * fig_col + c 
            if c == 0:
                ax[r, c].set_ylabel('current value')
            ax[r, c].set_xlabel(f'iter count(epoch {index})')
            ax[r, c].plot(np.array(epoch_current[index]))

    plt.show()

    for i in range(len(epoch_best)):
        print(f'best value {i}: {epoch_best[i][-1]}')


if __name__ == '__main__':
    sa_run()
