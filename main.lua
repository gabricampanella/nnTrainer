--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
--local DataLoader = require 'dataloader'
local models = require 'models/init'
local loader = require 'dataLoader'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

--Create model
local model, criterion = models.setup(opt)

--Load data and create batches
local trainData,valData,testData
if opt.test then
  trainData,valData,testData = loader.getData(opt)
else
  trainData,valData = loader.getData(opt)
end

local trainer = Trainer(model, criterion, opt, optimState)

--Training over all epochs
local bestValLoss=math.huge
for epoch = 1,opt.nEpochs do
  --Train single epoch
  local trainLoss,trainErr = trainer:train(epoch, trainData)
  --Run on validation set
  local valLoss,valErr = trainer:test(epoch, valData)
  --Check if best model
  local bestModel=false
  if valLoss<bestValLoss then
    bestModel=true
    bestValLoss=valLoss
  end
  print(('Epoch %d validation loss: %1.4f (best model validation loss: %1.4f)\n'):format(epoch,valLoss,bestValLoss))
  --Save model
  checkpoints.save(epoch,model,trainer.optimState,bestModel,opt)
  --Save loss for convergence plot
  if opt.task=='classification' then
    checkpoints.saveLoss(epoch,trainLoss,valLoss,trainErr,valErr,opt)
  elseif opt.task=='regression' then
    checkpoints.saveLoss(epoch,trainLoss,valLoss,opt)
  end
end

--Test data on best model
if opt.test then
  if opt.task =='classification' then
    local fd = io.open(paths.concat(opt.output,'test_predictions.csv'), 'w')
    fd:write('real,predicted\n')
    print('Test best model on test data: classification...')
    print('Loading best model from file: ' .. paths.concat(opt.output,'model_best.t7'))
    local bestModel=torch.load(paths.concat(opt.output,'model_best.t7'))
    local softMaxLayer = cudnn.SoftMax():cuda()
    bestModel:add(softMaxLayer)
    bestModel:cuda()
    bestModel:evaluate()
    print(('Predicting %d images...'):format(testData.target:size(1)))
    local pred=torch.FloatTensor(testData.target:size(1))
    for i=1,testData.target:size(1) do
      local img=testData.input[{{i},{},{},{}}]
      --local batch=img:view(1, table.unpack(img:size():totable()))
      local output=bestModel:forward(img:cuda()):squeeze()
      local prob, index = output:topk(1, true, true)
      fd:write(string.format('%u,%u\n',testData.target[i],index[1]))
      pred[i]=index[1]
    end
    fd:close()
    print('Predictions saved in file:' .. paths.concat(opt.output,'test_predictions.csv'))
    local accuracy=(testData.target-pred):eq(0):sum()/pred:size(1)
    print(('Overall classificacion accuracy: %1.4f'):format(accuracy))
  elseif opt.task=='regression' then
    local fd = io.open(paths.concat(opt.output,'test_predictions.csv'), 'w')
    fd:write('real,predicted\n')
    print('Test best model on test data: regression...')
    print('Loading best model from file: ' .. paths.concat(opt.output,'model_best.t7'))
    local bestModel=torch.load(paths.concat(opt.output,'model_best.t7'))
    bestModel:cuda()
    bestModel:evaluate()
    print(('Predicting %d images...'):format(testData.target:size(1)))
    local pred=torch.FloatTensor(testData.target:size(1))
    for i=1,testData.target:size(1) do
      local img=testData.input[{{i},{},{},{}}]
      local output=bestModel:forward(img:cuda()):squeeze()
      fd:write(string.format('%u,%u\n',testData.target[i],output))
      pred[i]=output
    end
    fd:close()
    print('Predictions saved in file:' .. paths.concat(opt.output,'test_predictions.csv'))
    local accuracy=(pred-testData.target):pow(2):sum()/pred:size(1)
    print(('Prediction MSE: %1.4f'):format(accuracy))
  end
end
