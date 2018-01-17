# reference http://districtdatalabs.silvrback.com/parameter-tuning-with-hyperopt
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from hpsklearn import HyperoptEstimator
# fmin是关键的函数，通过传入一个待优化的函数fn，以及一个参数空间space，使用一定的算法algo，在一定的迭代次数内查找最优值
best = fmin(fn=lambda x: x, space=hp.uniform('x', -1, 1), algo=tpe.suggest, max_evals=100)
print(best)


# function可以自定义，只要是一个有返回值的函数即可
def cus_f(x):
    if x == 0:
        return 1
    return np.sin(x) / x


x = np.linspace(-20, 20, 500)
plt.plot(x, [cus_f(i) for i in x])

best = fmin(fn=cus_f, space=hp.uniform('x', -20, 20), algo=tpe.suggest, max_evals=100)
print(best)

# 同时为了看到整个的黑盒过程，我们可以使用Trials对象进行观察
fspace = {
    'x': hp.uniform('x', -5, 5)
}


def f(params):
    x = params['x']
    val = x ** 2
    return {'loss': val, 'status': STATUS_OK}


trials = Trials()
best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=50, trials=trials)

print('best:', best)

print('trials:')
for trial in trials.trials:
    print(trial)



# 可以直接用来训练
