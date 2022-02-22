 # -*- coding: utf-8 -*-
"""
Training in the Source Domain
"""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf
import numpy as np
from Prius_model_new import Prius_model
import scipy.io as scio
import matplotlib.pyplot as plt
from Priority_Replay import Memory

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 1000      # The max episode
LR_A = 0.001             # The learning rate of the actor network
LR_C = 0.001             # The learning rate of the critic network
GAMMA = 0.9              # Reward discount
TAU = 0.01               # Soft replacement
MEMORY_CAPACITY = 50000  # Memory size
BATCH_SIZE = 64          # Batch size
RENDER = False

"""
DDPG module
Args:
    a_dim: the dimension of action space, 
    s_dim: the dimension of space space,
    a_bound: the boundary of action space value.
"""
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = Memory(capacity = MEMORY_CAPACITY)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # Build the actor network and actor target network
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        # Build the critic network and critic target network
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.td_error_up = abs(q_target - q)

        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        a_loss = - tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep = MAX_EPISODES)

        # Initialize OU noise
        self.ou_noise_prev = 0

    def NormalActionNoise(self, mu, sigma):
        return np.random.normal(mu, sigma)

    def OrnsteinUhlenbeckActionNoise(self, mu, sigma, theta=.15, dt=1e-2):
        ou_noise = self.ou_noise_prev + theta * (mu - self.ou_noise_prev) * dt + sigma * np.sqrt(dt) * np.random.normal(mu, sigma)
        self.ou_noise_prev = ou_noise
        return ou_noise

    def choose_action(self, s, loop, param_noise_scale, param_noise=True, action_noise_type=None):
        output_no_noise = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]   # 没有噪声

        # Add parameter noise
        if param_noise & (loop == 0):
            updates=[]
            for ea in self.ae_params:
                updates.append(tf.assign(ea, ea + tf.random_normal(tf.shape(ea), mean=0.,stddev=param_noise_scale)))
            self.sess.run(updates)
            output = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        else:
            output = output_no_noise

        # Add action noise
        if action_noise_type == 'gs':
            output_noise = output + self.NormalActionNoise(0, 0.05)     # Gaussian noise
        elif action_noise_type == 'ou':
            output_noise = output + self.OrnsteinUhlenbeckActionNoise(0, 0.15, 0.2, 0.01)      # OU noise
        elif action_noise_type == 'None':
            output_noise = output

        return output_noise, output_no_noise


    def learn(self):
        self.sess.run(self.soft_replace)
        tree_index, bt, ISWeight = self.memory.sample(BATCH_SIZE)
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
       
        abs_td_error = self.sess.run(self.td_error_up, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        self.memory.batch_update(tree_index, abs_td_error)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        self.memory.store(transition)
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        # Create actor network
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net1, 100, activation=tf.nn.relu, name = 'l2', trainable=trainable)
            net3 = tf.layers.dense(net2, 50, activation=tf.nn.relu, name = 'l3', trainable=trainable)
            a = tf.layers.dense(net3, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        # Create critic network
        with tf.variable_scope(scope):
            n_l1 = 200
            n_l2 = 100
            n_l3 = 50
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            w2 = tf.get_variable('w2', [n_l1, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
            w3 = tf.get_variable('w3', [n_l2, n_l3], trainable=trainable)
            b3 = tf.get_variable('b3', [1, n_l3], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.nn.relu(tf.matmul(net1, w2) + b2)
            net3 = tf.nn.relu(tf.matmul(net2, w3) + b3)
            return tf.layers.dense(net3, 1, trainable=trainable)  # Q(s,a)
    
    def savemodel(self):
        self.saver.save(self.sess, 'Checkpoints/source/save_net.ckpt', global_step = step_episode)

s_dim = 3
a_dim = 1
a_bound = 1
DDPG = DDPG(a_dim, s_dim, a_bound)
total_step = 0
step_episode = 0
mean_reward_all = 0
cost_Engine_list = []
cost_all_list = []
cost_Engine_100Km_list = []
mean_reward_list = []
std_reward_list = []
list_even = []
list_odd = []
mean_discrepancy_list = []
SOC_final_list = []

mu1 = 0
sigma1 = 0.03
threshold = 0.2
factor = 1.01

Prius = Prius_model()

for i in range(MAX_EPISODES):
    path = "Data_Standard Driving Cycles/Prius_source_data"
    path_list = os.listdir(path)
    random_data = np.random.randint(0,len(path_list))
    base_data = path_list[random_data]
    data = scio.loadmat(path + '/' + base_data)
    car_spd_one = data['speed_vector']
    total_milage = np.sum(car_spd_one) / 1000
    
    SOC = 0.65
    SOC_origin = SOC
    ep_reward = 0
    ep_reward_all = 0
    mileage = 0
    step_episode += 1
    SOC_data = []
    P_req_list = []
    Eng_spd_list = []
    Eng_trq_list = []
    Eng_pwr_list = []
    Eng_pwr_opt_list = []
    Gen_spd_list = []
    Gen_trq_list = []
    Gen_pwr_list = []
    Mot_spd_list = []
    Mot_trq_list = []
    Mot_pwr_list = []
    Batt_pwr_list = []
    inf_batt_list = []
    inf_batt_one_list = []
    Reward_list = []
    Reward_list_all = []
    T_list = []
    Mot_eta_list = []
    Gen_eta_list = []
    car_spd = car_spd_one[:, 0]
    car_a = car_spd_one[:, 0] - 0
    s = np.zeros(s_dim)
    s[0] = car_spd / 24.1683
    s[1] = (car_a - (-1.6114)) / (1.3034- (-1.6114))
    s[2] = SOC
    # set action_noise_type
    action_noise_type = 'None'
    # set param_noise_scale
    param_noise_scale = np.random.normal(mu1, sigma1)

    for j in range(car_spd_one.shape[1] - 1):
        print(str(i) + " ---> " + str(j) + "/", car_spd_one.shape[1])
        action, action_no_noise = DDPG.choose_action(s,j,param_noise_scale,False,action_noise_type)

        # adaptive parameter noise
        adaptive_policy_distance = np.sqrt(np.mean(np.square(action - action_no_noise)))
        if adaptive_policy_distance > threshold:
            param_noise_scale = param_noise_scale / factor
        else:
            param_noise_scale = param_noise_scale * factor

        a = np.clip(action, 0, 1)
        Eng_pwr_opt = (a[0]) * 56000
        
        out, cost, I = Prius.run(car_spd, car_a, Eng_pwr_opt, SOC)
        P_req_list.append(float(out['P_req']))
        Eng_spd_list.append(float(out['Eng_spd']))
        Eng_trq_list.append(float(out['Eng_trq'])) 
        Eng_pwr_list.append(float(out['Eng_pwr']))
        Eng_pwr_opt_list.append(float(out['Eng_pwr_opt']))
        Mot_spd_list.append(float(out['Mot_spd']))
        Mot_trq_list.append(float(out['Mot_trq']))        
        Mot_pwr_list.append(float(out['Mot_pwr']))  
        Gen_spd_list.append(float(out['Gen_spd']))
        Gen_trq_list.append(float(out['Gen_trq']))        
        Gen_pwr_list.append(float(out['Gen_pwr']))
        Batt_pwr_list.append(float(out['Batt_pwr']))   
        inf_batt_list.append(int(out['inf_batt']))
        inf_batt_one_list.append(int(out['inf_batt_one']))
        Mot_eta_list.append(float(out['Mot_eta']))
        Gen_eta_list.append(float(out['Gen_eta']))
        T_list.append(float(out['T'])) 
        SOC_new = float(out['SOC'])
        SOC_data.append(SOC_new)
        cost = float(cost)
        r = - cost
        ep_reward += r
        Reward_list.append(r)
        
        if SOC_new < 0.6 or SOC_new > 0.85:
            r = - ((350 * ((0.6 - SOC_new) ** 2)) + cost)
            
        car_spd = car_spd_one[:, j + 1]
        car_a = car_spd_one[:, j + 1] - car_spd_one[:, j]
        s_ = np.zeros(s_dim)   # [v, acc, SOC]
        s_[0] = car_spd / 24.1683   # velocity
        s_[1] = (car_a - (-1.6114)) / (1.3034- (-1.6114))   # acceleration
        s_[2] = SOC_new   # SoC
        DDPG.store_transition(s, a, r, s_)

        if total_step > MEMORY_CAPACITY:
            DDPG.learn()

        s = s_
        ep_reward_all += r
        Reward_list_all.append(r)

        total_step += 1
        SOC = SOC_new
        cost_Engine = - (ep_reward / 0.72 / 1000)   # The density of gasoline is 0.72.
        cost_all = - (ep_reward_all / 0.72 / 1000)

        if j == (car_spd_one.shape[1] - 2):
            SOC_final_list.append(SOC)
            mean_reward = np.mean(Reward_list_all)
            mean_reward_list.append(mean_reward)
            std_reward = np.std(Reward_list_all, ddof=1)
            std_reward_list.append(std_reward)
            cost_Engine += (SOC < SOC_origin) * (SOC_origin - SOC) * (201.6 * 6.5) * 3600 /(42600000) / 0.72
            cost_Engine_list.append(cost_Engine)
            cost_Engine_100Km_list.append(cost_Engine * (100 / (total_milage)))
            cost_all += (SOC < SOC_origin) * (SOC_origin - SOC) * (201.6 * 6.5) * 3600 /(42600000) / 0.72 
            cost_all_list.append(cost_all)
            print('Episode:', i, ' cost_Engine: %.3f' % cost_Engine, ' reward: %.3f' % -(ep_reward_all/100), ' SOC-final: %.3f' % SOC)

    DDPG.savemodel()

    SOC_final_arr = np.array(SOC_final_list)
    np.savetxt('./soc.txt', SOC_final_arr)
    cost_Engine_arr = np.array(cost_Engine_list)
    np.savetxt('./cost_Engine.txt', cost_Engine_arr)
    cost_all_arr = np.array(cost_all_list)
    np.savetxt('./cost_all.txt', cost_all_arr)
    mean_reward_arr = np.array(mean_reward_list)
    np.savetxt('./mean_reward.txt', mean_reward_arr)
    std_reward_arr = np.array(std_reward_list)
    np.savetxt('./std_reward.txt', std_reward_arr)
 
x = np.arange(0, len(SOC_data), 1)
y = SOC_data
plt.plot(x, y)
plt.xlabel('time')
plt.ylabel('SOC')