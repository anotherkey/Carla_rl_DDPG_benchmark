import tensorflow as tf
import gym
from tqdm import tqdm
import numpy as np
import os
import random
from carla.agent.agent import Agent
from carla.agent.data_process import DataProcess
from carla.carla_server_pb2 import Control


class DDPGAgent(Agent):
    def __init__(self, sess, actor, critic, gamma, replay_buffer=None, noise=None,
                 exploration_episodes=10000, max_episodes=10000, max_steps_episode=10000,
                 warmup_steps=5000, mini_batch=32, eval_episodes=10, eval_periods=100,
                 summary_dir=None, model_dir=None, detail=True, model_store_periods=1000, render_interval=50,
                 Inference_net_dir=None):
        """
        Deep Deterministic Policy Gradient Agent.
        Args:
            actor: actor network.
            critic: critic network.
            gamma: discount factor.
        """
        Agent.__init__(self)

        self.sess = sess
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.exploration_episodes = exploration_episodes
        self.max_episodes = max_episodes
        self.max_steps_episode = max_steps_episode
        self.warmup_steps = warmup_steps
        self.mini_batch = mini_batch
        self.eval_episodes = eval_episodes
        self.eval_periods = eval_periods

        self.summary_dir = summary_dir

        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.cur_episode = 0
        # if Inference is False:
        #     self.DP.load_model(sess)
        # else:
        self.Restore(Inference_net_dir)
        self.detail = detail
        self.model_dir = model_dir
        self.model_store_periods = model_store_periods
        self.train_t = 0
        self.render_interval = render_interval

    def Restore(self, net_dir):
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('models/saveNet_1907.ckpt-158700.meta')

        if not os.path.exists(net_dir):
            raise RuntimeError('failed to find the models path')
        ckpt = tf.train.get_checkpoint_state(net_dir)
        # self.saver.restore(self.sess, tf.train.latest_checkpoint('models/'))
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        print('Restoring from ', net_dir)

    def get_episode(self):
        return self.cur_episode


def fully_connected(inputs,
                    output_size,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    biases_initializer=tf.constant_initializer(0.0)):
    return tf.contrib.layers.fully_connected(inputs,
                                             output_size,
                                             activation_fn=activation_fn,
                                             weights_initializer=weights_initializer,
                                             weights_regularizer=weights_regularizer,
                                             biases_initializer=biases_initializer)


class ActorNetwork:
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, scope='Actor'):
        # super(ActorNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)
        self.sess = sess
        # self.sess.run(tf.global_variables_initializer())
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.action_bound = action_bound
        self.scope = scope
        self.DP = DataProcess(sess)


        # Actor network
        # self.inputs, self.phase, self.outputs, self.scaled_outputs = self.build_network('eval')
        self.inputs, self.outputs, self.scaled_outputs = self.build_network('eval')
        self.net_params = tf.trainable_variables(scope=self.scope)

        # Target network
        # self.target_inputs, self.target_phase, self.target_outputs, self.target_scaled_outputs = self.build_network(
        #     'target')
        self.target_inputs, self.target_outputs, self.target_scaled_outputs = self.build_network(
            'target')
        self.target_net_params = tf.trainable_variables(scope=self.scope)[len(self.net_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        self.update_target_bn_params = \
            [self.target_net_params[i].assign(self.net_params[i]) for i in range(len(self.target_net_params)) if
             self.target_net_params[i].name.startswith('BatchNorm')]

        # Combine dnetScaledOut/dnetParams with criticToActionGradient to get actorGradient
        # Temporary placeholder action gradient
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])

        self.actor_gradients = tf.gradients(self.outputs, self.net_params, -self.action_gradients)
        self.check_gradient = tf.gradients(self.outputs, self.net_params[-2])

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.net_params))

        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

        self.cur_action = {'steer': 0.0, 'acc': 0.0, 'brake': 0.0}
        self.next_action = {'steer': 0.0, 'acc': 0.0, 'brake': 0.0}
        self.Restore('models')

    def build_network(self, scope):
        inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        # phase = tf.placeholder(tf.bool)
        with tf.variable_scope(self.scope):
            with tf.variable_scope(scope):
                net = fully_connected(inputs, 256, activation_fn=tf.nn.relu)
                net = fully_connected(net, 256, activation_fn=tf.nn.relu)
                net = fully_connected(net, 128, activation_fn=tf.nn.relu)
                net = fully_connected(net, 64, activation_fn=tf.nn.relu)
                # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
                outputs = fully_connected(net, self.action_dim,
                                          activation_fn=tf.tanh)  # , weights_initializer=tf.random_uniform_initializer(-3e-5, 3e-5))
                scaled_outputs = tf.multiply(outputs,
                                             self.action_bound)  # Scale output to [-action_bound, action_bound]

        # return inputs, phase, outputs, scaled_outputs
        return inputs, outputs, scaled_outputs

    def train(self, *args):
        # args [inputs, action_gradients, phase]
        return self.sess.run(self.optimize, feed_dict={
            self.inputs: args[0],
            self.action_gradients: args[1],
            # self.phase: True
        })

    def check_(self, *args):
        grad_, outputs_ = self.sess.run([self.check_gradient, self.outputs], feed_dict={
            self.inputs: args[0],
            # self.phase: False
        })
        return [grad_[0], outputs_[0]]

    def predict(self, *args):
        return self.sess.run(self.scaled_outputs, feed_dict={
            self.inputs: args[0],
            # self.phase: False
        })

    def run_step(self, measurements, sensor_data, directions, target):

        steer = self.cur_action['steer']
        acc = self.cur_action['acc']
        brake = self.cur_action['brake']

        control = Control()
        feature_vector = self.DP.compute_feature(sensor_data)
        speed = measurements.player_measurements.forward_speed
        speed = speed / 10.0
        offroad = measurements.player_measurements.intersection_offroad
        other_lane = measurements.player_measurements.intersection_otherlane
        state = np.concatenate((feature_vector, (steer, acc - brake, speed, offroad, other_lane)))
        action = self.predict(np.expand_dims(state, 0))[0]

        control.steer = action[0]
        self.next_action['steer'] = action[0]
        if action[1] > 0:
            control.throttle = action[1]
            self.next_action['acc'] = action[1]
            control.brake = 0
            self.next_action['break'] = 0
        else:
            control.throttle = 0
            self.next_action['acc'] = 0
            control.brake = action[1]
            self.next_action['break'] = action[1]

        action_lambda = 0.5

        control.steer = self.next_action['steer'] * action_lambda + (1 - action_lambda) * self.cur_action['steer']
        control.throttle = self.next_action['acc'] * action_lambda + (1 - action_lambda) * self.cur_action['acc']
        control.brake = self.next_action['break'] * action_lambda + (1 - action_lambda) * self.cur_action['brake']
        control.hand_brake = 0
        control.reverse = 0

        self.cur_action['steer'] = control.steer
        self.cur_action['acc'] = control.throttle
        self.cur_action['brake'] = control.brake

        if measurements.player_measurements.forward_speed >= 8:
            control.throttle = 0.5 if control.throttle > 0.5 else control.throttle
        return control

    def predict_target(self, *args):
        return self.sess.run(self.target_scaled_outputs, feed_dict={
            self.target_inputs: args[0],
            # self.target_phase: False,
        })

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def Restore(self, net_dir):
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('models/saveNet_1907.ckpt-158700.meta')

        if not os.path.exists(net_dir):
            raise RuntimeError('failed to find the models path')
        ckpt = tf.train.get_checkpoint_state(net_dir)
        # self.saver.restore(self.sess, tf.train.latest_checkpoint('models/'))
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        print('Restoring from ', net_dir)


class CriticNetwork:
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, num_actor_vars,
                 scope='Critic'):
        # super(CriticNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.action_bound = action_bound
        self.scope = scope

        # Critic network
        # self.inputs, self.phase, self.action, self.outputs = self.build_network('eval')
        self.inputs, self.action, self.outputs = self.build_network('eval')
        self.net_params = tf.trainable_variables(scope=self.scope)

        # Target network
        # self.target_inputs, self.target_phase, self.target_action, self.target_outputs = self.build_network('target')
        self.target_inputs, self.target_action, self.target_outputs = self.build_network('target')

        self.target_net_params = tf.trainable_variables(self.scope)[len(self.net_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        self.update_target_bn_params = \
            [self.target_net_params[i].assign(self.net_params[i]) for i in range(len(self.target_net_params)) if
             self.target_net_params[i].name.startswith('BatchNorm')]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.outputs))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.net_params)

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.outputs, self.action)

    def build_network(self, scope):
        inputs = tf.placeholder(tf.float32, shape=[None, + self.state_dim])
        # phase = tf.placeholder(tf.bool)
        action = tf.placeholder(tf.float32, [None, self.action_dim])
        with tf.variable_scope(self.scope):
            with tf.variable_scope(scope):
                    net = fully_connected(inputs, 400, activation_fn=tf.nn.relu)
                    net = fully_connected(tf.concat([net, action], 1), 300, activation_fn=tf.nn.relu)
                    net = fully_connected(net, 128, activation_fn=tf.nn.relu)
                    outputs = fully_connected(net,
                                              1)  # , weights_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))

        # return inputs, phase, action, outputs
        return inputs, action, outputs

    def train(self, *args):
        # args (inputs, action, predicted_q_value, phase)
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
            # self.phase: True
        })

    def predict(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            # self.phase: False
        })

    def predict_target(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: args[0],
            self.target_action: args[1],
            # self.target_phase: False
        })

    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action,
            # self.phase: False
        })

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)
