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
local checkpoint = {}

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.save(epoch, model, optimState, isBestModel, opt)
   -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- create a clean copy on the CPU without modifying the original network
   model = deepCopy(model):float():clearState()

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'

   torch.save(paths.concat(opt.output, modelFile), model)
   torch.save(paths.concat(opt.output, optimFile), optimState)
--   torch.save(paths.concat(opt.output, 'latest.t7'), {
--      epoch = epoch,
--      modelFile = modelFile,
--      optimFile = optimFile,
--   })

   if isBestModel then
      torch.save(paths.concat(opt.output, 'model_best.t7'), model)
   end
end

function checkpoint.saveLoss(epoch,trainLoss,valLoss,trainErr,valErr,opt)
  local filename=paths.concat(opt.output, 'lossConvergence.csv')
  if opt.task=='classification' then
    if epoch==1 then
      local fd=io.open(filename,'w')
      fd:write('epoch,trainLoss,valLoss,trainErr,valErr\n')
      fd:write(string.format('%u,%6.5f,%6.5f,%1.4f,%1.4f\n',epoch,trainLoss,valLoss,trainErr,valErr))
      fd:close()
    else
      local fd=io.open(filename,'a')
      fd:write(string.format('%u,%6.5f,%6.5f,%1.4f,%1.4f\n',epoch,trainLoss,valLoss,trainErr,valErr))
      fd:close()
    end
  elseif opt.task=='regression' then
    if epoch==1 then
      local fd=io.open(filename,'w')
      fd:write('epoch,trainLoss,valLoss\n')
      fd:write(string.format('%u,%6.5f,%6.5f\n',epoch,trainLoss,valLoss))
      fd:close()
    else
      local fd=io.open(filename,'a')
      fd:write(string.format('%u,%6.5f,%6.5f\n',epoch,trainLoss,valLoss))
      fd:close()
    end
  end
end
return checkpoint
