--  Gabriele 2/13/2017
--  Adapted from Facebook:
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Gabriele\'s Torch-7 ResNet Training script')
   cmd:text('Adapted from Facebook\'s framework')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:text('------------ General options --------------------')
   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-nChannels',       3,         'Number of channels in images')
   cmd:option('-output', '', 'Path to output directory.')
   cmd:option('-test', 'false', 'Whether or not to test on test set after training.')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-nClasses',         0,      'Number of classes in the dataset')
   ---------- Optimization options ----------------------
   cmd:text('---------- Optimization options ----------------------')
   cmd:option('-LR',              0.1,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   ---------- Model options ----------------------------------
   cmd:text('---------- Model options ----------------------------------')
   cmd:option('-model',      'resnet', 'Options: resnet | alexnet')
   cmd:option('-task', 'classification', 'Options: classification | regression')
   ---------- Resnet options ----------------------------------
   cmd:text('---------- Resnet options ----------------------------------')
   cmd:option('-depth',        34,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-shortcutType', '',       'ResNet Shortcut types. Options: A | B | C')
   ---------- Preprocessing options ----------------------------------
   cmd:text('---------- Preprocessing options --------------------------------')
   cmd:option('-centerCrop',  'false', 'Whether or not to center crop')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.test = opt.test ~= 'false'
   opt.centerCrop = opt.centerCrop ~= 'false'

   return opt
end

return M
