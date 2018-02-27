""" cSGD and cMSGD

Tensorflow implementations of the adaptives algorithms in
Li, Qianxiao, Tai, Cheng, and E, Weinan. Stochastic modified
equations and adaptive stochastic gradient algorithms.
In International Conference on Machine Learning, pp.
2101–2110, 2017.
URL: <http://www.proceedings.mlr.press/v70/li17f/li17f.pdf>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""


import tensorflow as tf


def _copy_var(variable, suffix, trainable=False,
              initializer=tf.zeros_initializer()):
    """Copy variable with shape, dtype, and name+suffix"""
    shape = variable.get_shape()
    name = variable.name.partition(':')[0]
    dtype = variable.dtype
    name = name+'/'+suffix
    return tf.get_variable(
        name, shape, dtype, trainable=trainable, initializer=initializer)


def _ema_mean(mean, value, beta):
    """Create EMA mean update op

    This is the standard exponential moving average:
        mean <- beta*mean + (1-beta)*value
    """
    return tf.assign_sub(
        mean, (1-beta)*(mean-value))


def _ema_var(var, mean, value, beta):
    """Create EMA variance update op

    Uses the standard exponential moving variance/covariance formula
    See e.g.
        Schubert, E.; Weiler, M.; Kriegel, H. P. (2014). SigniTrend:
        scalable detection of emerging topics in textual streams by
        hashed significance thresholds. Proceedings of the 20th ACM
        SIGKDD international conference on Knowledge discovery and
        data mining - KDD '14. pp. 871–880.
    """
    return tf.assign_sub(
        var, (1-beta)*(var - beta*(value-mean)**2))


def _ema_covar(covar, mean1, value1, mean2, value2, beta):
    """Create EMA covariance update op

    Uses the standard exponential moving variance/covariance formula
    See e.g.
        Schubert, E.; Weiler, M.; Kriegel, H. P. (2014). SigniTrend:
        scalable detection of emerging topics in textual streams by
        hashed significance thresholds. Proceedings of the 20th ACM
        SIGKDD international conference on Knowledge discovery and
        data mining - KDD '14. pp. 871–880.
    """
    return tf.assign_sub(
        covar, (1-beta)*(covar - beta*(value1-mean1)*(value2-mean2)))


class CSGDOptimizer(object):
    """cSGD optimizer class"""
    def __init__(self, learning_rate=1.0, beta_min=0.9, beta_max=0.99,
                 eta_max=0.1, name='cSGD'):
        """Initializer

        Args:
            learning_rate (float, optional): Defaults to 1.0.
                Initial learning rate scale
            beta_min (float, optional): Defaults to 0.9.
                Minimum EMA decay rate
            beta_max (float, optional): Defaults to 0.99.
                Maximum EMA decay rate
            eta_max (float, optional): Defaults to 0.1.
                Maximum learning rate. Should make this as big as possible
                without training blowing up, i.e. if plain SGD doesn't blow
                up at a particular lr, it is ually safe to set eta_max to
                that value
            name (string, optional): Defaults to 'cSGD'.
                Name of the instance (for variable suffix)
        """
        self._learning_rate = learning_rate
        self._eta_max = eta_max
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._name = name

    def _set_ema_vars(self, var_list):
        """Create a list of dicts of EMA variables"""
        eps = 0.0
        dicts = []
        beta_init = self._beta_min  # Small init decay to forget init condition
        for v in var_list:
            d = {
                'g': _copy_var(v, 'g'),
                'g_var': _copy_var(
                    v, 'g_var',
                    initializer=tf.constant_initializer(eps)),
                'xi': _copy_var(v, 'xi'),
                'xi_var': _copy_var(
                    v, 'xi_var',
                    initializer=tf.constant_initializer(eps)),
                'g_xi_var': _copy_var(
                    v, 'g_xi_var',
                    initializer=tf.constant_initializer(eps)),
                'beta': _copy_var(
                    v, 'beta', initializer=tf.constant_initializer(beta_init)),
                'u': _copy_var(
                    v, 'u',
                    initializer=tf.constant_initializer(self._learning_rate))}
            dicts.append(d)
        return dicts

    def _update_ema(self, ema, g, v):
        """Create ops to update EMA variables
        Note: accumulates mean and (co)variances, instead of
        mean and mean-squares, as outlined in the original
        paper.
        """
        var_ops = [
            _ema_var(ema['g_var'], ema['g'], g, ema['beta']),
            _ema_var(ema['xi_var'], ema['xi'], v, ema['beta']),
            _ema_covar(ema['g_xi_var'], ema['g'], g,
                       ema['xi'], v, ema['beta'])]
        with tf.control_dependencies(var_ops):
            mean_ops = [
                _ema_mean(ema['g'], g, ema['beta']),
                _ema_mean(ema['xi'], v, ema['beta'])]
        return tf.group(*mean_ops)

    def _update_beta(self, ema):
        """Create ops to update EMA decay rate"""
        g_var_normalized = ema['g_var'] / (ema['g']**2 + ema['g_var'])
        g_var_normalized = tf.clip_by_value(
            g_var_normalized, 0, 1)
        new_beta = self._beta_min + \
            (self._beta_max - self._beta_min) * \
            g_var_normalized
        beta_update_op = tf.assign(ema['beta'], new_beta)
        return beta_update_op

    def _update_u(self, ema):
        """Create ops to update learning rate scale
        Uses accumulated variances instead of mean-squares
        to improve stability.
        """
        a = ema['g_xi_var'] / ema['xi_var']
        sigma = ema['g_var']
        m = 0.5*(ema['g']**2 + ema['g_var']) / a
        new_u = 2*m / (self._eta_max * sigma)
        all_ones = tf.ones_like(new_u)
        new_u = tf.minimum(new_u, all_ones)
        new_u = tf.where(new_u > 0, new_u, all_ones)
        u_update_op = _ema_mean(ema['u'], new_u, ema['beta'])
        return u_update_op

    def _update_params(self, ema, g, v):
        """Create ops to update trainable parameters"""
        return tf.assign_sub(v, self._eta_max * ema['u'] * g)

    def _train_ops(self, ev_list, grad_list, var_list):
        """Create cSGD training/updating ops"""
        ops = []
        for ev, grad, var in zip(ev_list, grad_list, var_list):
            # EMA update ops
            ema_op = self._update_ema(ev, grad, var)

            # Train
            train_op = self._update_params(ev, grad, var)

            # Update u and beta
            with tf.control_dependencies([ema_op, train_op]):
                u_op = self._update_u(ev)
                with tf.control_dependencies([u_op]):
                    beta_op = self._update_beta(ev)

            ops.append(beta_op)
        return ops

    def minimize(self, cost, var_list=None):
        """Create train op

        Args:
            cost (tf tensor): loss tensor
            var_list (list of tf tensor, optional): Defaults to None.
                list of trainable variables. if None, sets to
                tf.trainable_variables()

        Returns:
            tf op: cSGD train op
        """

        if var_list is None:
            var_list = tf.trainable_variables()
        grad_list = tf.gradients(cost, var_list)
        with tf.variable_scope(self._name):
            self._ema_vars = self._set_ema_vars(var_list)
        self._train_ops = self._train_ops(self._ema_vars, grad_list, var_list)
        return tf.group(*self._train_ops, name=self._name+'_train_step')


class CMSGDOptimizer(CSGDOptimizer):
    """cMSGD optimizer class"""
    def __init__(self, learning_rate=0.1, momentum=0.9,
                 beta_min=0.9, beta_max=0.99, name='cMSGD'):
        """Initializer

        Args:
            learning_rate (float, optional): Defaults to 0.1.
                Initial learning rate
            learning_rate (float, optional): Defaults to 0.9.
                Initial momentum parameter
            beta_min (float, optional): Defaults to 0.9.
                Minimum EMA decay rate
            beta_max (float, optional): Defaults to 0.999.
                Maximum EMA decay rate
            name (string, optional): Defaults to 'cMSGD'.
                Name of the instance (for variable suffix)
        """
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._name = name

    def _set_ema_vars(self, var_list):
        """Create ops to update EMA variables"""
        dicts = []
        # beta_init = 0.5*(self._beta_min + self._beta_max)
        beta_init = self._beta_min
        for v in var_list:
            d = {
                'g': _copy_var(v, 'g'),
                'g_var': _copy_var(v, 'g_var'),
                'xi': _copy_var(v, 'xi'),
                'xi_var': _copy_var(v, 'xi_var'),
                'g_xi_var': _copy_var(v, 'g_xi_var'),
                'beta': _copy_var(
                    v, 'beta', initializer=tf.constant_initializer(beta_init)),
                'mu': _copy_var(
                    v, 'mu',
                    initializer=tf.constant_initializer(self._momentum)),
                'mom': _copy_var(v, 'momentum')}
            dicts.append(d)
        return dicts

    def _update_mu(self, ema):
        """Create ops to update momentum parameters"""
        a = ema['g_xi_var'] / ema['xi_var']
        sigma = ema['g_var']
        m = 0.5*(ema['g']**2 + ema['g_var']) / a
        factor1 = 1 - 2*tf.sqrt(self._learning_rate*tf.abs(a))
        factor1 = tf.nn.relu(factor1)
        factor2 = 1 - 0.25*self._learning_rate*sigma / m
        factor2 = tf.nn.relu(factor2)
        new_mu = tf.minimum(factor1, factor2)
        all_ones = tf.ones_like(new_mu)
        new_mu = tf.where(a > 0, new_mu, all_ones)
        mu_update_op = _ema_mean(ema['mu'], new_mu, ema['beta'])
        return mu_update_op

    def _update_params(self, ema, g, v):
        """Create ops to update trainable parameters"""
        update_mom = tf.assign(
            ema['mom'], ema['mu']*ema['mom'] - self._learning_rate*g)
        with tf.control_dependencies([update_mom]):
            update_param = tf.assign_add(v, ema['mom'])
        return update_param

    def _train_ops(self, ev_list, grad_list, var_list):
        """Create cMSGD training/updating ops"""
        ops = []
        for ev, grad, var in zip(ev_list, grad_list, var_list):
            # EMA update ops
            ema_op = self._update_ema(ev, grad, var)

            # Train
            train_op = self._update_params(ev, grad, var)

            # Update u and beta
            with tf.control_dependencies([ema_op, train_op]):
                mu_op = self._update_mu(ev)
                with tf.control_dependencies([mu_op]):
                    beta_op = self._update_beta(ev)

            ops.append(beta_op)
        return ops
