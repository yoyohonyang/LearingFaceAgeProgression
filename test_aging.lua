require 'torch'
require 'image'
require 'nn'
require 'cutorch'


require 'FaceAging.DataLoader_aging_test'
require 'FaceAging.ShaveImage'
require 'FaceAging.TotalVariation'
require 'FaceAging.InstanceNormalization'


local utils = require 'FaceAging.utils'
local preprocess = require 'FaceAging.preprocess'


local cmd = torch.CmdLine()

-- Model options
cmd:option('-model', '.models/CACD_Aging.t7')
cmd:option('-h5_file', 'data/CACD/CACD_test.h5')
cmd:option('-use_instance_norm', 1)
cmd:option('-preprocessing', 'vgg')
cmd:option('-gpu', 1)
cmd:option('-backend', 'cuda')
cmd:option('-output_dir', './CACD_aging_results')
cmd:option('-output_filename', 'aging_results')
cmd:option('-num_val_batches', 25)
cmd:option('-batch_size', 8)


local function main()
	local opt = cmd:parse(arg)
	paths.mkdir(opt.output_dir)
	-- Figure out preprocessing
	if not preprocess[opt.preprocessing] then
		local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
		error(string.format(msg, opt.preprocessing))
	end
	preprocess = preprocess[opt.preprocessing]
	local loader = DataLoader_aging_test(opt)
	local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
	cutorch.setDevice(1)
	-- Set up the generator
	local netG = nil
	if opt.model ~= '' then
		print('Loading model from ' .. opt.model)
		netG = torch.load(opt.model).netG:type(dtype)
	else
		print('No such file!')  
	end
	loader:reset('val')
	--- Warm up the GPU
	if opt.use_cudnn then
		print('Warm up the GPU...')
		for t1 = 1, 10 do
			local x1 = torch.randn(opt.batch_size, 3, 224, 224):type(dtype)
			local y1 = netG:forward(x1)
		end
	end
	--- Testing
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
end

main()