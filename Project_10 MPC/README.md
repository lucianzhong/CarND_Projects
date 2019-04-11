Model Predictive Controller Project

Introduction
 	The goal of this project is to navigate a track in a Udacity-provided simulator, which communicates telemetry and track waypoint data via websocket, by sending steering and acceleration commands back to the simulator. The solution must be robust to 100ms latency, as one may encounter in real-world application.

This solution, as the Nanodegree lessons suggest, makes use of the IPOPT and CPPAD libraries to calculate an optimal trajectory and its associated actuation commands in order to minimize error with a third-degree polynomial fit to the given waypoints. The optimization considers only a short duration's worth of waypoints, and produces a trajectory for that duration based upon a model of the vehicle's kinematics and a cost function based mostly on the vehicle's cross-track error (roughly the distance from the track waypoints) and orientation angle error, with other cost factors included to improve performance.

模型预测控制（MPC）是一种致力于将更长时间跨度、甚至于无穷时间的最优化控制问题，分解为若干个更短时间跨度，或者有限时间跨度的最优化控制问题，并且在一定程度上仍然追求最优解。模型预测控制由如下三个要素组成：
	预测模型：预测模型能够在短时间内很好地预测系统状态的变化
	在线滚动优化：通过某种最优化算法来优化未来一段短时间的控制输入，使得在这种控制输入下预测模型的输出与参考值的差距最小
	反馈校正：到下一个时间点根据新的状态重新进行预测和优化

透过一些Sensor fusion我们可以把Camera/Lidar/Radar的资料转换成道路和物体的座标，配合汽车的物理模型(汽车加速度，加速度，角速度)我们就可以来做MPC了。在每一个时间点t，我们可以用汽车现在的状态来估算下一个时间t1点的状态x, y, psi, v, delta, a
x, y 是汽车的座标，psi 是汽车面向的方向，v是汽车的瞬时速度而delta是汽车轮胎的角度，a是则是汽车的加速率（油门/煞车)。因次我们可以用以下的公式来推测下一个时间点汽车的状态（公式如下）。其中最特别的就是第三行，Lf 是一个常数，代表汽车前端到汽车重心的长度。意思就是车子不会只依照着方向盘的角度转向，车子的长度和车子移动的速度也会影响单位时间车子改变方向的量

通过使用不同的预测模型和损失函数，可以构造出各种模型预测控制器，但是，总的来说，模型预测控制往往可以分解成如下几步： 
1. 从 tt 时刻开始，预测未来 aa 步系统的输出信号 
2. 基于模型的控制信号以及相应的输出信号，构造损失函数，并且通过调整控制信号最优化损失函数 
3. 将控制信号输入到系统 
4. 等到下一个时间点，在新的状态重复步骤1

x1 = x0 + v0 * cos(psi) * dt; 
y1 = y0 + v0 * sin(psi) * dt; 
psi1 = psi0 + v0 / Lf * delta * dt; // Lf = 2.67 in simulator 
v1 = v0 + a0 * dt;
有了这些公式我们就可以预测出未来T (prediction horizo​​n) 的时间内汽车可能的路径。因此把一个follow的问题转化成一个控制最佳化的问题。尝试各种控制的排列组合[delta0, a0, delta1, a1, delta2, a2 …]找出最合适的路径

ipopt的C++ library来计算来解最佳路径，因此我们只要定义cost function 然后把限制丢进去ipopt就可以了，超方便！首先cost想要minimize的就是error，error包括偏移路径的量(cte)和汽车的方向的偏移(epsi)。另外为了让汽车稳定要尽量减少使用方向盘跟油门的(actuator)，同时也要避免突然踩油门煞车或是突然转方向盘(actuator change)，最后还要避免电脑出奥步(直接把车开超慢甚至停在路中间就没有error了)所以再加上速度太慢的cost就完成
