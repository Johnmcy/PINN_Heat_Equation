
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
from network import Network

# 定义一个类，用于实现PINN(Physics-informed Neural Networks)
class PINN:
    # 构造函数
    def __init__(self):
        # 选择使用GPU还是CPU
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 定义神经网络
        self.model = Network(
            input_size=2,  # 输入层神经元数
            hidden_size=16,  # 隐藏层神经元数
            output_size=1,  # 输出层神经元数
            depth=8,  # 隐藏层数
            act=torch.nn.Tanh  # 输入层和隐藏层的激活函数
        ).to(device)  # 将这个神经网络存储在GPU上（若GPU可用）

        self.h = 0.1  # 设置空间步长
        self.k = 0.1  # 设置时间步长
        x = torch.arange(-1, 1 + self.h, self.h)  # 在[-1,1]区间上均匀取值，记为x
        t = torch.arange(0, 1 + self.k, self.k)  # 在[0,1]区间上均匀取值，记为t

        # 将x和t组合，形成时间空间网格，记录在张量X_inside中
        self.X_inside = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T

        # 边界处的时空坐标
        bc1 = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T  # x=-1边界
        bc2 = torch.stack(torch.meshgrid(x[-1], t)).reshape(2, -1).T  # x=+1边界
        ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T  # t=0边界
        self.X_boundary = torch.cat([bc1, bc2])  # 将所有边界处的时空坐标点整合为一个张量
        self.T_boundary = ic

        # 边界处的u值
        u_ic = 1.0 + torch.sin(ic[:, 0])  # t=0边界处采用第一类边界条件usin(x)
        self.U_T_boundary = u_ic  # 将所有边界处的u值整合为一个张量=1 + sin(x)
        self.U_T_boundary = self.U_T_boundary.unsqueeze(1)

        # 将数据拷贝到GPU
        self.X_inside = self.X_inside.to(device)
        self.X_boundary = self.X_boundary.to(device)
        self.T_boundary = self.T_boundary.to(device)
        self.U_T_boundary = self.U_T_boundary.to(device)
        self.X_inside.requires_grad = True  # 设置：需要计算对X_inside的梯度
        self.X_boundary.requires_grad = True  # 设置：需要计算对X_boundary的梯度

        # 设置准则函数为MSE，方便后续计算MSE
        self.criterion = torch.nn.MSELoss()

        # 定义迭代序号，记录调用了多少次loss
        self.iter = 1

        # 设置lbfgs优化器
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=0.1,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

        # 设置adam优化器
        self.adam = torch.optim.Adam(self.model.parameters())

    # 损失函数
    def loss_func(self):
        # 将导数清零
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        # 第一部分loss：初始值loss
        T_pred_boundary = self.model(self.T_boundary)  # 使用当前模型计算u在边界处的预测值
        loss_boundary_T = self.criterion(
            T_pred_boundary, self.U_T_boundary)  # 计算边界处的MSE

        # 第二部分loss:内点非物理产生的loss
        U_inside = self.model(self.X_inside)  # 使用当前模型计算内点处的预测值

        # 使用自动求导方法得到U对X的导数
        du_dX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=U_inside,
            grad_outputs=torch.ones_like(U_inside),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dx = du_dX[:, 0]  # 提取对第x的导数
        du_dt = du_dX[:, 1]  # 提取对第t的导数

        # 使用自动求导方法得到U对X的二阶导数
        du_dxx = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=du_dX,
            grad_outputs=torch.ones_like(du_dX),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]
        loss_equation = self.criterion(
            du_dt, 0.1 * du_dxx)  # 计算物理方程的MSE

        # 第三部分loss: 边界诺伊曼条件产生的误差
        U_X_boundary = self.model(self.X_boundary)

        du_dX = torch.autograd.grad(
            inputs=self.X_boundary,
            outputs=U_X_boundary,
            grad_outputs=torch.ones_like(U_X_boundary),
            retain_graph=True,
            create_graph=True
        )[0]

        du_dx = du_dX[:, 0]  # 计算边界梯度不为0带来的损失
        loss_equation_X = self.criterion(du_dx, 0 * du_dx)

        # 最终的loss由三项组成
        loss = loss_equation + loss_boundary_T + loss_equation_X

        # loss反向传播，用于给优化器提供梯度信息
        loss.backward()

        # 每计算100次loss在控制台上输出消息
        if self.iter % 100 == 0:
            print(self.iter, loss.item())
            losses.append(loss.item())
        self.iter = self.iter + 1
        return loss

    # 训练
    def train(self):
        self.model.train()  # 设置模型为训练模式

        # 首先运行5000步Adam优化器
        print("采用Adam优化器")
        for i in range(3000):
            self.adam.step(self.loss_func)
        # 然后运行lbfgs优化器
        print("采用L-BFGS优化器")
        self.lbfgs.step(self.loss_func)


if __name__ == '__main__':

    # 实例化PINN
    pinn = PINN()

    # 开始训练
    losses = []  # 记录训练误差利于绘图
    pinn.train()

    # 将模型保存到文件
    torch.save(pinn.model, 'model.pth')

    # 训练误差可视化
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iter // 100")
    plt.ylabel("Training Loss")
    plt.title('Training Error vs Epochs')
    plt.show()
