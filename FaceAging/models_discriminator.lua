require 'nn'
require 'nngraph'

require 'FaceAging.ShaveImage'
require 'FaceAging.TotalVariation'
require 'FaceAging.InstanceNormalization'


local D = {}

function D.build_model()
  local netD = nn.Sequential()
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
  return netD
end

return D

