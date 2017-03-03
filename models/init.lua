--  Gabriele 3/3/2017
--  Adapted from Facebook
--
-- Generic model loader

require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt)
   local model
   
   if opt.finetune ~= '' then
   
      print('=> Loading pre-trained model from file: models/' .. opt.finetune)
      model = torch.load(opt.finetune)
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')
      local orig = model:get(#model.modules)
      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()
      model:remove(#model.modules)
      model:add(linear:type('torch.CudaTensor'))
   
   else
      print('=> Creating model from file: models/' .. opt.model .. '.lua')
      model = require('models/' .. opt.model)(opt)

      -- First remove any DataParallelTable
      if torch.type(model) == 'nn.DataParallelTable' then
         model = model:get(1)
      end

      -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
      -- containers override backwards to call backwards recursively on submodules
      if opt.shareGradInput then
         print('Sharing grad inputs')
         M.shareGradInput(model)
      end

      -- Set the CUDNN flags
      if opt.cudnn == 'fastest' then
         cudnn.fastest = true
         cudnn.benchmark = true
      elseif opt.cudnn == 'deterministic' then
         -- Use a deterministic convolution implementation
         model:apply(function(m)
            if m.setMode then m:setMode(1, 1, 1) end
         end)
      end

      -- Wrap the model with DataParallelTable, if using more than one GPU
      if opt.nGPU > 1 then
         local gpus = torch.range(1, opt.nGPU):totable()
         local fastest, benchmark = cudnn.fastest, cudnn.benchmark

         local dpt = nn.DataParallelTable(1, true, true)
            :add(model, gpus)
            :threads(function()
               local cudnn = require 'cudnn'
               cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end)
         dpt.gradInput = nil

         model = dpt:cuda()
      end
   end

   local criterion
   if opt.task == 'classification' then
      if opt.model == 'alexnet' then
         criterion = nn.ClassNLLCriterion():cuda()
      elseif opt.model == 'resnet' then
         criterion = nn.CrossEntropyCriterion():cuda()
      end
   elseif opt.task == 'regression' then
      criterion = nn.MSECriterion():cuda()
   else
      print('Bad task')
   end

   return model, criterion
end

function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end

return M
