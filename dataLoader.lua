--Gabriele 2/15/2017
--loads data and divides it into batches
local t = require 'transforms'
local M={}

local function preprocess(data,opt)
  if opt.centerCrop then
    return t.Compose{
      t.ColorNormalize(opt.meanstd,opt.nChannels),
      t.CenterCrop(224)
    }(data)
  else
    return t.Compose{
      t.ColorNormalize(opt.meanstd,opt.nChannels)
    }(data)
  end
end

function M.getData(opt)

  local function makeBatch(input,target,size,batchSize)
    local perm = torch.randperm(size)
    local dataLoad={}
    for i=1,size,batchSize do

      local indices=perm:narrow(1,i,math.min(batchSize,size-i+1))
      local batch
      local batchTarget
      if opt.task=='classification' then
        batchTarget=torch.IntTensor(math.min(batchSize,size-i+1))
      elseif opt.task=='regression' then
        batchTarget=torch.FloatTensor(math.min(batchSize,size-i+1))
      end
      for ii,j in ipairs(indices:totable()) do
        local inp=input:narrow(1,j,1)[1]
        inp=preprocess(inp,opt)
        imageSize=inp:size():totable()
        if not batch then
          batch=torch.FloatTensor(math.min(batchSize,size-i+1),table.unpack(imageSize))
        end
        batch[ii]:copy(inp)
        batchTarget[ii]=target[j]
      end
      --table.remove(imageSize, 1)
      --batch=batch:view(batchSize,table.unpack(imageSize))
      table.insert(dataLoad,{input=batch,target=batchTarget})
    end
    return dataLoad
  end

  --Load data
  local fname=opt.data
  print('Loading file ' .. fname .. '\n')
  local data=torch.load(fname)
  --make train data
  local input=data.train.data
  local target=data.train.labels
  --get mean and sd
  local mean={}
  local std={}
  for i=1,opt.nChannels do
    mean[i] = input[{ {}, {i}, {}, {}  }]:mean()
    std[i] = input[{ {}, {i}, {}, {}  }]:std()
  end
  opt.meanstd={
    mean=mean,
    std=std
  }
  local batchSize=opt.batchSize
  local size=input:size(1)
  local nbatches=math.ceil(size/batchSize)

  print('Training dataset info-----')
  print(string.format('  %u images',size))
  print(string.format('  %u channels',input:size(2)))
  print(string.format('  Image size: %u x %u',input:size(3),input:size(4)))
  print('--------------------------')
  print(string.format('  batch size: %u',batchSize))
  print(string.format('  number of batches: %u',nbatches))
  print('--------------------------')
  print('Creating batches...')
  trainLoad=makeBatch(input,target,size,batchSize)
  print('Done\n')

  --make validation data
  local input=data.val.data
  local target=data.val.labels
  local batchSize=opt.batchSize
  local size=input:size(1)
  local nbatches=math.ceil(size/batchSize)

  print('Validation dataset info-----')
  print(string.format('  %u images',size))
  print(string.format('  %u channels',input:size(2)))
  print(string.format('  Image size: %u x %u',input:size(3),input:size(4)))
  print('--------------------------')
  print(string.format('  batch size: %u',batchSize))
  print(string.format('  number of batches: %u',nbatches))
  print('--------------------------')
  print('Creating batches...')
  valLoad=makeBatch(input,target,size,batchSize)
  print('Done\n')

  --if testing on test data
  if opt.test then
    local input=data.test.data
    local target=data.test.labels
    local size=input:size(1)

    print('Test dataset info-----')
    print(string.format('  %u images',size))
    print(string.format('  %u channels',input:size(2)))
    print(string.format('  Image size: %u x %u',input:size(3),input:size(4)))
    print('--------------------------')
    print('Creating data...')
    
    local testLoad
    for i=1,size do
      local inp=input[{{i},{},{},{}}]
      inp=preprocess(inp:view(input:size(2),input:size(3),input:size(4)),opt)
      local imageSize=inp:size():totable()
      if not testLoad then
        testLoad={
          input=torch.FloatTensor(size,table.unpack(imageSize)),
          target=torch.FloatTensor(size)
        }
      end
      testLoad.input[i]:copy(inp)
      testLoad.target[i]=target[i]
    end
    print('Done\n')

    return trainLoad,valLoad,testLoad

  else
    return trainLoad,valLoad
  end

end

return M
