--Many thanks to Justin Johnson (http://cs.stanford.edu/people/jcjohns/) for sharing the code

require 'torch'
require 'nn'
local cjson = require 'cjson'


local M = {}



function M.setup_gpu(gpu, backend, use_cudnn)
  local dtype = 'torch.FloatTensor'
  if gpu >= 0 then
    if backend == 'cuda' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(gpu + 1)
      dtype = 'torch.CudaTensor'
      if use_cudnn then
        require 'cudnn'
        cudnn.benchmark = true
      end
    elseif backend == 'opencl' then
      require 'cltorch'
      require 'clnn'
      cltorch.setDevice(gpu + 1)
      dtype = torch.Tensor():cl():type()
      use_cudnn = false
    end
  else
    use_cudnn = false
  end
  return dtype, use_cudnn
end




return M

