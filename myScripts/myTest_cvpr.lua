require 'torch'
require 'optim'
require 'image'
require 'nn'
require 'cutorch'

require 'FaceAging.DataLoader_aging'
require 'FaceAging.ShaveImage'
require 'FaceAging.TotalVariation'
require 'FaceAging.InstanceNormalization'


local utils = require 'FaceAging.utils'
local preprocess = require 'FaceAging.preprocess'
local models_aging = require 'FaceAging.models_aging'
local models_discriminator = require 'FaceAging.models_discriminator'


local opt = {
    resume_from_checkpoint = "./checkpoints/CACD_gan_3D2_20180521_01_5fold5_aging/CACD_gan_3D2_20180521_01_5fold5_aging.t7" ,
	use_instance_norm = 1,
    h5_file = 'data/CACD/CACD_disorder_5fold2_agegroup0_new.h5',
    padding_type= 'reflect-start',
    preprocessing = 'vgg',  
    gpu = 1,      
    backend =  'cuda', 'cuda|opencl', 
    checkpoints_dir = './checkpoints',
    checkpoint_name = 'test_20181203_CACD_gan_3D2_20181203_01_5fold1_aging_chazhi',
    num_val_batches = 200,
	n_cls = 3,	
	batch_size = 8,
	max_train = -1,
}

paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.checkpoint_name)
 
 -- Figure out preprocessing
if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
end
preprocess = preprocess[opt.preprocessing]
local loader = DataLoader_aging(opt)


local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
cutorch.setDevice(1)

-- set up generator
local netG = nil
if opt.resume_from_checkpoint ~= '' then
    print('Loading checkpoint from ' .. opt.resume_from_checkpoint)
    netG = torch.load(opt.resume_from_checkpoint).netG:type(dtype)
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

   local x1 = torch.randn(opt.batch_size, 6, 224, 224):type(dtype)
   local y1 = netG:forward(x1)

   local dy = torch.randn(#y1):type(dtype)
   -- netG:updateGradInput(x1, dy) 
   netG:backward(x1,dy) 
   return errG,  gradParametersG
end


---Warm up the GPUs
print('Warm up the GPU...')
for t1 = 1, 10 do
	optim.adam(fGx, parametersG, optimStateG)
end


netG:evaluate()
local val_age_groups = 7
for t2 = 1, opt.num_val_batches do
    print(t2)
	image_out_final = nil
	local x = loader:getBatch('val')	    		
    x = x:type(dtype)
	local val_target = torch.FloatTensor(opt.batch_size,opt.n_cls,224,224):fill(0):type(dtype)
		
	for j = 1, val_age_groups do
	
        
		local val_target = torch.FloatTensor(opt.batch_size,opt.n_cls,224,224):fill(0):type(dtype)
		if j == 1 then 
			val_target[{{},1,{},{}}] = 1
		elseif j == 2 then
			val_target[{{},1,{},{}}] = 0.67
			val_target[{{},2,{},{}}] = 0.33
		elseif j == 3 then
			val_target[{{},1,{},{}}] = 0.33
			val_target[{{},2,{},{}}] = 0.67		
		elseif j == 4 then
			val_target[{{},2,{},{}}] = 1	   
		elseif j == 5 then
			val_target[{{},2,{},{}}] = 0.67
			val_target[{{},3,{},{}}] = 0.33		
		elseif j == 6 then
			val_target[{{},2,{},{}}] = 0.33
			val_target[{{},3,{},{}}] = 0.67	   
		elseif j == 7 then		
			val_target[{{},3,{},{}}] = 1	   
		end
			
        local x_targetAge = torch.cat(x,val_target,2)		
	    local out = netG:forward(x_targetAge)
	
		out = preprocess.deprocess(out)
		input = preprocess.deprocess(x)
		image_out = nil
		for i2 = 1, out:size(1) do
            if image_out==nil then image_out = torch.cat(input[i2]:float(),out[i2]:float(),3)			 
            else image_out = torch.cat(image_out, torch.cat(input[i2]:float(),out[i2]:float(),3), 2) end
        end
		
		if image_out_final==nil then image_out_final = image_out		 
		else image_out_final = torch.cat(image_out_final, image_out:narrow(3,225,224) , 3) end	
		end	
		
	image.save(paths.concat(opt.checkpoints_dir,  opt.checkpoint_name , t2 .. '_train_res.png'), image_out_final)   
end



