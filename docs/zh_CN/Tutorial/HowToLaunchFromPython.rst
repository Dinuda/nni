如何从 Python 发起实验
===========================================

概述
--------
从 ``nni v2.0`` 起，我们提供了一种全新方式发起 Experiment 。 在此之前，您需要在 yaml 文件中配置实验，然后使用 ``nnictl``命令启动 Experiment 。 现在，您还可以直接在python文件中配置和运行 Experiment 。 如果您熟悉 Python 编程，那么无疑会为您带来很多便利。

如何使用
----------
成功安装 ``nni`` 之后，您可以通过以下3个步骤使用 Python 脚本开始 Experiment 。

..

    步骤1 - 初始化 Tuner


.. code-block:: python

    from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
    tuner = HyperoptTuner('tpe')

很简单，您已经成功初始化了一个名为 ``tuner`` 的 ``HyperoptTuner`` 实例。

查看 NNI 所有的 `内置 Tuner <../builtin_tuner.rst>`__。

..

    步骤2 - 初始化并配置 Experiment 实例

.. code-block:: python

    experiment = Experiment(tuner=tuner, training_service='local')

现在，您已经在上一步中初始化了带有 Tuner 的 ``Experiment`` 实例，由于 ``training_service ='local'`` ，此实验将在本地计算机上运行。

查看 NNI 支持的所有 `训练平台 <../training_services.rst>`__。

.. code-block:: python

    experiment.config.experiment_name = 'test'
    experiment.config.trial_concurrency = 2
    experiment.config.max_trial_number = 5
    experiment.config.search_space = search_space
    experiment.config.trial_command = 'python3 mnist.py'
    experiment.config.trial_code_directory = Path(__file__).parent
    experiment.config.training_service.use_active_gpu = True

使用类似 ``experiment.config.foo ='bar'`` 的形式来配置您的 Experiment 。

参阅不同平台所需的 `参数配置 <../reference/experiment_config.rst>`__。

..

    步骤3 - 运行

.. code-block:: python

    experiment.run(port=8081)

现在，您已经成功启动了 NNI Experiment。 您可以在浏览器中输入``localhost:8081`` 以实时观察实验。

示例
-------
以下是这种新的启动方法的示例。 你可以在 :githublink:`mnist-tfv2/launch.py <examples/trials/mnist-tfv2/launch.py>` 找到实验代码。

.. code-block:: python

    from pathlib import Path
    from nni.experiment import Experiment
    from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

    tuner = HyperoptTuner('tpe')

    search_space = {
        "dropout_rate": { "_type": "uniform", "_value": [0.5, 0.9] },
        "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
        "hidden_size": { "_type": "choice", "_value": [124, 512, 1024] },
        "batch_size": { "_type": "choice", "_value": [16, 32] },
        "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
    }

    experiment = Experiment(tuner, 'local')
    experiment.config.experiment_name = 'test'
    experiment.config.trial_concurrency = 2
    experiment.config.max_trial_number = 5
    experiment.config.search_space = search_space
    experiment.config.trial_command = 'python3 mnist.py'
    experiment.config.trial_code_directory = Path(__file__).parent
    experiment.config.training_service.use_active_gpu = True

    experiment.run(8081)

API
---

..  autoclass:: nni.experiment.Experiment
    :members:
