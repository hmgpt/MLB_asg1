import gymnasium as gym
import numpy as np
import pygame
import pygame.font
import time

# 初始化 Pygame
pygame.init()
font = pygame.font.Font(None, 36)  # 创建字体对象

env = gym.make("Taxi-v3", render_mode="rgb_array")  # 使用 rgb_array 模式获取图像
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 超参数
alpha = 0.2  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_min = 0.01
epsilon_decay = 0.9995
num_episodes = 20000  # 增加训练轮数

# 训练过程
for episode in range(num_episodes):
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # 调整 epsilon

    total_reward = 0
    observation, info = env.reset()
    terminated = False

    while not terminated:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(q_table[observation])  # 利用

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 更新 Q 表
        old_value = q_table[observation, action]
        next_max = np.max(q_table[next_observation])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[observation, action] = new_value

        observation = next_observation

        if terminated or truncated:
            break

    if episode % 1000 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# 测试经过训练的智能体
env = gym.make("Taxi-v3", render_mode="rgb_array")
observation, info = env.reset()
terminated = False

print("\nTraining finished. Now testing...")

# 初始化 Pygame 窗口
screen = pygame.display.set_mode((env.render().shape[0] + 200, env.render().shape[1] ))  # 交换宽度和高度，增加文本空间
pygame.display.set_caption("Taxi-v3 with Pygame Visualizer")

total_rewards = 0

# 游戏循环
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = np.argmax(q_table[observation])  # 总是利用策略
    next_observation, reward, terminated, truncated, info = env.step(action)
    total_rewards += reward

    # 获取游戏图像并旋转
    frame = env.render()
    frame_surface = pygame.surfarray.make_surface(frame)
    rotated_frame = pygame.transform.rotate(frame_surface, 270)  # 逆时针旋转 90 度

    # 绘制游戏画面
    screen.fill((0, 0, 0))  # 清屏
    screen.blit(rotated_frame, (0, 0))  # 显示旋转后的图像

    # 绘制文本
    text_color = (255, 255, 255)  # 文本颜色
    text_position = (10, env.render().shape[0] )  # 调整文本显示位置以适配旋转后的图像

    current_state_text = font.render(f"State: {observation}", True, text_color)
    screen.blit(current_state_text, text_position)

    action_text = font.render(f"Action: {action}", True, text_color)
    screen.blit(action_text, (text_position[0], text_position[1] + 40))

    reward_text = font.render(f"Reward: {reward}", True, text_color)
    screen.blit(reward_text, (text_position[0], text_position[1] + 80))

    total_reward_text = font.render(f"Total Rewards: {total_rewards}", True, text_color)
    screen.blit(total_reward_text, (text_position[0], text_position[1] + 120))

    pygame.display.flip()  # 更新显示

    observation = next_observation

    # 如果任务完成或达到时间限制，重置环境
    if terminated or truncated:
        observation, info = env.reset()

    clock.tick(1)  # 控制帧率

# 清理
env.close()
pygame.quit()
