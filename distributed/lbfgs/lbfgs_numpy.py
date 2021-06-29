# Copyright (C) 2021 NumS Development Team.
# Copyright (C) 2021 Intel corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List, Union

import numpy as np


def forward(app, X, theta):
    #print("forward:", X.shape, theta.shape)
    eta = X @ theta
    mu = 1.0 / (1.0 + app.exp(-eta))
    return mu


def objective(app, y, mu):
    one = 1.0
    return - app.sum(y * app.log(mu) + (one - y) * app.log(one - mu + 1e-14))


def grad(X, y, mu):
    return X.T @ (mu - y)


def bt_linesearch(app,
                  X, y, theta,
                  grad, p,
                  rho=1.e-1, init_alpha=1.0, c=1e-4, min_alpha=1e-10):

    def f(theta_prime):
        return objective(app, y, forward(app, X, theta_prime))

    alpha = init_alpha
    f_val = f(theta)
    f_next = f(theta + alpha * p)
    while app.isnan(f_next) or f_next > f_val + c * alpha * grad.T @ p:
        alpha *= rho
        if alpha < min_alpha:
            return min_alpha
        # print("btls step alpha=%s" % alpha)
        f_next = f(theta + alpha * p)
    return alpha


class LBFGSMemory(object):

    def __init__(self, k, s, y):
        self.k = k
        self.s = s
        self.y = y
        ys_inner = s.T @ y
        self.rho = 1.0 / ys_inner
        self.gamma = ys_inner / y.T @ y


class LBFGS(object):

    def __init__(self, app,
                 m=3, max_iter=100,  thresh=1e-5, dtype=np.float32):
        self.app = app
        self.m = m
        self.max_iter = max_iter
        self.thresh = thresh
        self.dtype = dtype
        self.k = 0
        self.identity = None
        self.memory: Union[List[LBFGSMemory], List[None]] = [None]*m

    def get_H(self):
        if self.k == 0:
            return self.identity
        else:
            mem: LBFGSMemory = self.memory[-1]
            assert mem.k == self.k-1
            return mem.gamma * self.identity

    def get_p(self, H, g):
        q = g
        forward_vars = []
        for i in range(-1, -self.m-1, -1):
            mem_i: LBFGSMemory = self.memory[i]
            if mem_i is None:
                break
            alpha = mem_i.rho * mem_i.s.T @ q
            q -= alpha * mem_i.y
            forward_vars.insert(0, (alpha, mem_i))
        r = H @ q
        for alpha, mem_i in forward_vars:
            beta = mem_i.rho * mem_i.y.T @ r
            r += mem_i.s * (alpha - beta)
        return r

    def execute(self, X, y, theta):

        if self.k != 0:
            raise Exception("Unexpected state.")

        #print("exec", (X.shape[1], X.shape[1]), self.dtype)
        self.identity = self.app.eye(X.shape[1], dtype=self.dtype)

        # TODO: Try sampling a new block every iteration.
        # TODO: Try stochastic approach, given line search...
        # X_btls = X[:X.block_shape[0]]
        # y_btls = y[:y.block_shape[0]]
        X_btls = X
        y_btls = y

        g = grad(X, y, forward(self.app, X, theta))
        next_g = None
        next_theta = None
        while self.k < self.max_iter:
            H = self.get_H()
            p = - self.get_p(H, g)
            init_alpha = min(1.0, 10**(self.k-self.max_iter/2))
            alpha = bt_linesearch(self.app, X_btls, y_btls,
                                  theta, g, p,
                                  rho=1e-2,
                                  init_alpha=init_alpha,
                                  c=1e-4,
                                  min_alpha=1e-30)
            #print("alpha", alpha)
            # print("alpha", alpha,
            #       "objective", f(theta).get(),
            #       "grad_norm", self.app.sqrt(g.T @ g).get())
            next_theta = theta + alpha * p
            if self.k + 1 >= self.max_iter:
                # Terminate immediately if this is the last iteration.
                theta = next_theta
                break
            next_g = grad(X, y, forward(self.app, X, next_theta))
            theta_diff = next_theta - theta
            grad_diff = next_g - g
            mem: LBFGSMemory = LBFGSMemory(k=self.k, s=theta_diff, y=grad_diff)
            self.memory.append(mem)
            self.memory.pop(0)
            self.k += 1
            theta = next_theta
            g = next_g
            # if self.converged(next_g):
            #     break

        # Reset vars.
        self.k = 0
        self.identity = None
        self.memory: Union[List[LBFGSMemory], List[None]] = [None]*self.m

        return theta

    def converged(self, g):
        return self.app.sqrt(g.T @ g) < self.thresh


def logistic(app, X, y, max_iter, m):
    Xc = app.concatenate([X, app.ones((X.shape[0], 1), dtype=X.dtype)],
                         axis=1,
    )
    theta = app.zeros((Xc.shape[1],), dtype=Xc.dtype)
    lbfgs_optimizer = LBFGS(app, m=m, max_iter=max_iter, dtype=Xc.dtype)
    theta = lbfgs_optimizer.execute(Xc, y, theta)
    return forward(app, Xc, theta)


def sample_set(app):
    #block_shape = (100, 10)
    #rs = app.random.RandomState(1337)
    #X1 = rs.normal(loc=5.0, size=shape, 
    #               #block_shape=block_shape
    #              )
    #X2 = rs.normal(loc=10.0, size=shape,
    #               #block_shape=block_shape
    #              )
    X = np.loadtxt("X.csv", delimiter=',')
    y = np.loadtxt("y.csv", delimiter=',')
    #print("sample", X.shape, y.shape)
    return X, y


def run_lbfgs():
    start_time = time.time()
    X, y = sample_set(np)
    
    y_pred_proba = logistic(np, X, y, max_iter=10, m=3)
    print("scheduling submitted.")
    y_pred = (y_pred_proba > 0.5).astype(np.float32)
    print("prediction submitted.")
    error = (np.sum(np.abs(y - y_pred)) / X.shape[0]).astype(np.float32)
    total_time = time.time() - start_time

    print("opt", "lbfgs")
    print("total time", total_time)
    print("error (1-accuracy)", error)
    # print("norm", model.grad_norm_sq(X, y).get())
    # print("objective", model.objective(X, y).get())
    return total_time
