使用 NNI Tuners 自动进行模型压缩
========================================

使用 NNI 能轻松实现自动模型压缩

首先，使用 NNI 压缩模型
---------------------------------

可使用 NNI 轻松压缩模型。 Take pruning for example, you can prune a pretrained model with L2FilterPruner like this

.. code-block:: python

   from nni.algorithms.compression.pytorch.pruning import L2FilterPruner
   config_list = [{ 'sparsity': 0.5, 'op_types': ['Conv2d'] }]
   pruner = L2FilterPruner(model, config_list)
   pruner.compress()

The 'Conv2d' op_type stands for the module types defined in :githublink:`default_layers.py <nni/compression/pytorch/default_layers.py>` for pytorch.

Therefore ``{ 'sparsity': 0.5, 'op_types': ['Conv2d'] }``\ means that **all layers with specified op_types will be compressed with the same 0.5 sparsity**. 当调用 ``pruner.compress()`` 时，模型会通过掩码进行压缩。随后还可以微调模型，此时 **被剪除的权重不会被更新**。

然后，进行自动化
-------------------------

The previous example manually chose L2FilterPruner and pruned with a specified sparsity. Different sparsity and different pruners may have different effects on different models. This process can be done with NNI tuners.

Firstly, modify our codes for few lines

.. code-block:: python

    import nni
    from nni.algorithms.compression.pytorch.pruning import *
   
    params = nni.get_parameters()
    sparsity = params['sparsity']
    pruner_name = params['pruner']
    model_name = params['model']

    model, pruner = get_model_pruner(model_name, pruner_name, sparsity)
    pruner.compress()

    train(model)  # your code for fine-tuning the model
    acc = test(model)  # test the fine-tuned model
    nni.report_final_results(acc)

Then, define a ``config`` file in YAML to automatically tuning model, pruning algorithm and sparsity.

.. code-block:: yaml

    searchSpace:
    sparsity:
      _type: choice
      _value: [0.25, 0.5, 0.75]
    pruner:
      _type: choice
      _value: ['slim', 'l2filter', 'fpgm', 'apoz']
    model:
      _type: choice
      _value: ['vgg16', 'vgg19']
    trainingService:
    platform: local
    trialCodeDirectory: .
    trialCommand: python3 basic_pruners_torch.py --nni
    trialConcurrency: 1
    trialGpuNumber: 0
    tuner:
      name: grid

The full example can be found :githublink:`here <examples/model_compress/pruning/config.yml>`

Finally, start the searching via

.. code-block:: bash

   nnictl create -c config.yml
