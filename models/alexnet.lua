--  Gabriele 3/3/2017
--  Alexnet implementation

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local nClasses = opt.nClasses
   local nChannels = opt.nChannels

   -- Configurations for AlexNet:
   print(' | AlexNet-')

   -- The ResNet ImageNet model
   local features = nn.Sequential()
   features:add(Convolution(nChannels,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(cudnn.ReLU(true))
   features:add(Max(3,3,2,2))                   -- 55 ->  27
   features:add(Convolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(cudnn.ReLU(true))
   features:add(Max(3,3,2,2))                   --  27 ->  13
   features:add(Convolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(Convolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(Convolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(Max(3,3,2,2))                   -- 13 -> 6

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, nClasses))
   if opt.task =='classification' then
      classifier:add(nn.LogSoftMax())
   end

   local model = nn.Sequential()
   model:add(features):add(classifier)

  
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
