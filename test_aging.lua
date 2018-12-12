require 'torch'
require 'optim'
require 'image'
require 'nn'
require 'cutorch'


require 'FaceAging.DataLoader_aging_test'
require 'FaceAging.ShaveImage'
require 'FaceAging.TotalVariation'
require 'FaceAging.InstanceNormalization'


local utils = require 'FaceAging.utils'
local preprocess = require 'FaceAging.preprocess'




local opt = {
    models = "./models/CACD_Aging.t7" ,
	use_instance_norm = 1,
    h5_file = 'data/CACD/CACD_test.h5',
    preprocessing = 'vgg',  
    gpu = 1,      
    backend =  'cuda', 'cuda|opencl', 
    output_dir = './test_CACD_20181212',
    output_filename = 'test_CACD_20181212',
    num_val_batches = 20,
	batch_size = 8,
}

paths.mkdir(opt.output_dir)

 
 -- Figure out preprocessing
if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
end
preprocess = preprocess[opt.preprocessing]
local loader = DataLoader_aging_test(opt)


local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
cutorch.setDevice(2)

-- set up generator
local netG = nil
if opt.models ~= '' then
    print('Loading checkpoint from ' .. opt.models)
    netG = torch.load(opt.models).netG:type(dtype)
else
    print('No such file!')  
end


loader:reset('val')
local optimStateG = {learningRate = 1e-10}
local parametersG, gradParametersG = netG:getParameters()
---Warm up the GPUs
local fGx = function(x)   
    gradParametersG:zero()
	errG = 0

   local x1 = torch.randn(opt.batch_size, 3, 224, 224):type(dtype)
   local y1 = netG:forward(x1)

   local dy = torch.randn(#y1):type(dtype)
   -- netG:updateGradInput(x1, dy) 
   netG:backward(x1,dy) 
   return errG,  gradParametersG
end


---Warm up the GPU
if opt.use_cudnn then
print('Warm up the GPU...')
for t1 = 1, 10 do
	optim.adam(fGx, parametersG, optimStateG)
end
end

netG:evaluate()
print('Testing...')
for t2 = 1, opt.num_val_batches do

    print(t2)
	image_out_final = nil
	local x = loader:getBatch('val')	    		
    x = x:type(dtype)
	local out = netG:forward(x)
	
	out = preprocess.deprocess(out)
	input = preprocess.deprocess(x)
	image_out = nil
	for i2=1, out:size(1) do
        if image_out==nil then image_out = torch.cat(input[i2]:float(),out[i2]:float(),3)			 
        else image_out = torch.cat(image_out, torch.cat(input[i2]:float(),out[i2]:float(),3), 2) end
    end
	image.save(paths.concat(opt.output_dir,  opt.output_filename ..'_'.. t2 .. '_test_res.png'), image_out)
end




