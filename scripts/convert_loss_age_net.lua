require 'torch'
require 'optim'
require 'image'
require 'nn'
require 'cutorch'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'
local models_aging = require 'fast_neural_style.models_aging'

local opt = {
    use_instance_norm = 1,
    task = 'aging',
    tanh_constant = 150, 
    preprocessing = 'vgg',
    prev_dim = 256,
    arch = 'R256,R256,R256,R256,R256,u128,u64,u32,c9s1-3',
    pixel_loss_type =  'L2', 'L2|L1|SmoothL1',
    pixel_loss_weight = 1.0,
    percep_age_loss_weight =  15.0,
	percep_id_loss_weight =  0.01,
    tv_strength = 1e-6,
    gpu = 1,      
    use_cudnn= 0,
    backend =  'cuda', 'cuda|opencl', 
-- Optimization
    num_iterations = 30000,
    max_train = -1,
    batch_size = 8,
    learning_rate = 1e-3,
    lr_decay_every = -1,
    lr_decay_factor =  0.5,
    weight_decay = 0,
-- Checkpointing
    checkpoints_dir = './checkpoints',
    checkpoint_name = 'loss_age_net_gan_morph',
    checkpoint_every = 200,
    num_val_batches = 1,

-- Options for Age Loss networks  (At this moment, it is the same as the vgg features networks)
	loss_age_net = './models/loss_age_net_gan_morph.t7',
    age_weights = {'1.0','1.0','1.0','1.0'},
    age_layers = {'4','9','16','23'},
	loss_id_net = 'models/vgg_face_torch/VGG_FACE.t7',
	id_weights = {'1.0','1.0','1.0','1.0'},
    id_layers = {'4','9','16','23'},

-- Options for perceived age loss 
	use_target_age = -1,
	target_age = 70,
	age_target = 'age_target.jpg'
}

paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.checkpoint_name)

if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
end
preprocess = preprocess[opt.preprocessing]

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
cutorch.setDevice(4)
--local loss_age_net = torch.load(opt.loss_age_net).model:type(dtype)
local loss_age_net = torch.load(opt.loss_age_net):type(dtype)
print('loss_age_net:')
print(loss_age_net)


--[[new_loss_age_net = nn.Sequential()


local s_1 = nn.Sequential()
for i = 1,4 do
   local temp_layer = nil
   temp_layer = loss_age_net.modules[i]
   s_1:add(temp_layer)
end

new_loss_age_net:add(s_1)



local s_2 = nn.Sequential()
for i = 5,9 do
   local temp_layer = nil
   temp_layer = loss_age_net.modules[i]
   s_2:add(temp_layer)
end
local cc_2 = nn.ConcatTable()
cc_2:add(nn.Identity())
cc_2:add(s_2)
cc_2:add(s_2)
cc_2:add(s_2)
new_loss_age_net:add(cc_2)



pp_3 =  nn.ParallelTable()
s_3 = nn.Sequential()
for i = 10,16 do
   local temp_layer = nil
   temp_layer = loss_age_net.modules[i]
   s_3:add(temp_layer)
end
pp_3:add(nn.Identity())
pp_3:add(nn.Identity())
pp_3:add(s_3)
pp_3:add(s_3)
new_loss_age_net:add(pp_3)

pp_4 =  nn.ParallelTable()
s_4 = nn.Sequential()
for i = 17,23 do
   local temp_layer = nil
   temp_layer = loss_age_net.modules[i]
   s_4:add(temp_layer)
end
pp_4:add(nn.Identity())
pp_4:add(nn.Identity())
pp_4:add(nn.Identity())
pp_4:add(s_4)
new_loss_age_net:add(pp_4)


print('new_loss_age_net:')
print(new_loss_age_net)

]]--
local age_target_img = image.load(opt.age_target, 3, 'float')
age_target_img = image.scale(age_target_img, 224)
local H, W = age_target_img:size(2), age_target_img:size(3)
age_target_img = preprocess.preprocess(age_target_img:view(1, 3, H, W)):type(dtype)
--local output = new_loss_age_net:forward(age_target_img)
local age_features = loss_age_net:forward(age_target_img)
print(age_features)


--[[if use_cudnn then
    cudnn.convert(new_loss_age_net, nn)
end
new_loss_age_net:float()

filename = string.format('%s.t7', opt.checkpoint_name)
torch.save(paths.concat(opt.checkpoints_dir, opt.checkpoint_name , filename), new_loss_age_net)
]]--


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

---------build up the discriminator

netD  = nn.Sequential() 
local pt = nn.ParallelTable()
  -- define x_f_1
  -- state size 64*224*224
local x_f_1 = nn.Sequential() 
x_f_1:add(nn.SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
x_f_1:add(nn.SpatialBatchNormalization(128)):add(nn.LeakyReLU(0.2, true))
x_f_1:add(nn.SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
x_f_1:add(nn.SpatialBatchNormalization(256)):add(nn.LeakyReLU(0.2, true))
x_f_1:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
x_f_1:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
x_f_1:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))
x_f_1:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
-- state size 512*14*14
x_f_1:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))
x_f_1:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
x_f_1:add(nn.SpatialConvolution(512, 1, 4, 4, 2, 2, 1, 1))
 
  -- define x_f_2
  -- state size 128*112*112
local x_f_2 = nn.Sequential()   
x_f_2:add(nn.SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
x_f_2:add(nn.SpatialBatchNormalization(256)):add(nn.LeakyReLU(0.2, true))
x_f_2:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
x_f_2:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
x_f_2:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))
x_f_2:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
-- state size 512*14*14
x_f_2:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))
x_f_2:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
x_f_2:add(nn.SpatialConvolution(512, 1, 4, 4, 2, 2, 1, 1))


  -- define x_f_3
  -- state size 256*56*56
local x_f_3 = nn.Sequential() 
 
x_f_3:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
x_f_3:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
x_f_3:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))
x_f_3:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
-- state size 512*14*14
x_f_3:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))
x_f_3:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
x_f_3:add(nn.SpatialConvolution(512, 1, 4, 4, 2, 2, 1, 1))
 

  -- define x_f_4
  -- state size 512*28*28
local x_f_4 = nn.Sequential() 
x_f_4:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))
x_f_4:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
-- state size 512*14*14
x_f_4:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))
x_f_4:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
x_f_4:add(nn.SpatialConvolution(512, 1, 4, 4, 2, 2, 1, 1))
 
pt:add(x_f_1)
pt:add(x_f_2)
pt:add(x_f_3)
pt:add(x_f_4)
netD:add(pt)
netD:add(nn.JoinTable(1,3))
  
  
netD:add(nn.SpatialBatchNormalization(4)):add(nn.LeakyReLU(0.2, true))
  
local ct = nn.ConcatTable()
local full_feature_1 = nn.Sequential() 
full_feature_1:add(nn.SpatialConvolution(4, 4, 3, 3, 1, 1, 1, 1))
local full_feature_2 = nn.Sequential() 
full_feature_2:add(nn.SpatialConvolution(4, 4, 3, 3, 1, 1, 0, 0))
full_feature_2:add(nn.View(-1):setNumInputDims(3))
full_feature_2:add(nn.LogSoftMax())
  --state size batch_size*4
ct:add(full_feature_1)
ct:add(full_feature_2)
netD:add(ct)
 
netD:apply(weights_init)
netD:type(dtype)
print(netD)

output = netD:forward(age_features)
print('output:')
print(output)

nll = nn.ClassNLLCriterion():type(dtype)
mse = nn.MSECriterion():type(dtype)
pc = nn.ParallelCriterion():add(mse):add(nll,0.5)

print(pc)
local real_label = 1
local target1 = torch.FloatTensor(output[1]:size()):fill(real_label):type(dtype)
print(target1)
local target2 = torch.IntTensor{2}:type(dtype)
print(target2)
target = {target1,target2}
print(target)

loss1= mse:forward(output[1],target1)
loss2= nll:forward(output[2],target2)
loss = pc:forward(output,target)
print(loss)
local df_do = pc:backward(output,target)

local df_do_img = netD:updateGradInput(age_features, df_do)
local df_dg = loss_age_net:updateGradInput(age_target_img, df_do_img)
print('#df_do')
print(df_do)
print('#df_do_img')
print(df_do_img)
print('#df_dg')
print(#df_dg)

