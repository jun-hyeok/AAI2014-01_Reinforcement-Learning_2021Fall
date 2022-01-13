import numpy as np


def q_from_v(env, V, s, gamma=0.99):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, s_next, reward in env.MDP[s][a]:
            q[a] += prob * (reward + gamma * V[s_next])
    return q


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            q = q_from_v(env, V, s, gamma)
            v = np.sum(policy[s] * q)
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)
        a = np.argmax(q)
        policy[s] = np.eye(env.nA)[a]
    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    # 1. initialization
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        # 2. policy evaluation
        V = policy_evaluation(env, policy, gamma, theta)
        # 3. policy improvement
        new_policy = policy_improvement(env, V, gamma)
        if np.all(policy == new_policy):
            # if policy-stable
            break
        policy = new_policy.copy()
    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        delta = 0
        for s in range(env.nS):
            q = q_from_v(env, V, s, gamma)
            v = np.max(q)
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        if delta < theta:
            policy = policy_improvement(env, V, gamma)
            break
    return policy, V
