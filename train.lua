--  Gabriele 2/15/2017
--  Adapted from Facebook:
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, trainData)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = #trainData
   local lossSum=0.0
   local miserrSum=0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in ipairs(trainData) do
      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      local miserr=self:computeScore(output, sample.target, self.opt)
      miserrSum = miserrSum + miserr*batchSize

      print((' | Epoch: [%d][%d/%d] Batch time %.3f Loss crit. %1.4f Misclass. error %1.4f LR %1.4f'):format(
         epoch, n, trainSize, timer:time().real, loss, miserr, self.optimState.learningRate))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()

   end

   return lossSum / N, miserrSum / N
end

function Trainer:test(epoch, valData)

   local timer = torch.Timer()
   local size = #valData
   local lossSum=0.0
   local miserrSum=0.0
   local N = 0

   self.model:evaluate()
   for n, sample in ipairs(valData) do

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      local miserr=self:computeScore(output, sample.target, self.opt)
      miserrSum = miserrSum + miserr*batchSize

      print((' | Test: [%d][%d/%d] Batch time %.3f  Loss crit. %1.4f Misclass. error %1.4f'):format(
         epoch, n, size, timer:time().real, loss, miserr))

      timer:reset()

   end
   self.model:training()

   print((' * Finished epoch # %d\n'):format(
      epoch))

   return lossSum / N, miserrSum / N
end

function Trainer:computeScore(output, target, opt)
  if opt.task=='classification' then
    local batchSize=output:size(1)
    local _,predictions=output:float():sort(2,true)
    predictions=predictions:narrow(2,1,1)
    local correct=predictions:eq(target:long():view(batchSize, 1))
    return 1.0 - (correct:sum() / batchSize)
  elseif opt.task=='regression' then
    return
  end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   decay = math.floor((epoch - 1) / 30)
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
