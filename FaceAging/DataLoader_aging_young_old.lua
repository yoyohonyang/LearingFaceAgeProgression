require 'torch'
require 'hdf5'

local utils = require 'FaceAging.utils'
local preprocess = require 'FaceAging.preprocess'
local DataLoader_aging_young_old = torch.class('DataLoader_aging_young_old')


--[[
split: (1)train_young; (2)train_old; (3)val.
]]--

function DataLoader_aging_young_old:__init(opt)
  assert(opt.h5_file_young, 'Must provide h5_file_young')
  assert(opt.h5_file_old, 'Must provide h5_file_old')
  assert(opt.batch_size, 'Must provide batch size')
  self.preprocess_fn = preprocess[opt.preprocessing].preprocess
  self.task = opt.task
  
  self.h5_file_young = hdf5.open(opt.h5_file_young, 'r')
  self.h5_file_old = hdf5.open(opt.h5_file_old, 'r')
  self.batch_size = opt.batch_size
  
  self.split_idxs = {
    train_young = 1,
	train_old = 1,
    val = 1,
  }
  
  self.image_paths = {
    train_young_x = '/train_x/images',
    train_old_x = '/train_x/images',	
    val_x = '/val_x/images',	
  }
  
  local train_size_young = self.h5_file_young:read(self.image_paths.train_young_x):dataspaceSize()
  local train_size_old = self.h5_file_old:read(self.image_paths.train_old_x):dataspaceSize()

  self.split_sizes = {
    train_young = train_size_young[1],
	train_old = train_size_old[1],
    val = self.h5_file_young:read(self.image_paths.val_x):dataspaceSize()[1],  
  }
  self.num_channels = train_size_young[2]
  self.image_height = train_size_young[3]
  self.image_width = train_size_young[4]

  self.num_minibatches = {}
  for k, v in pairs(self.split_sizes) do
    self.num_minibatches[k] = math.floor(v / self.batch_size)
  end
  
  self.rgb_to_gray = torch.FloatTensor{0.2989, 0.5870, 0.1140}
end


function DataLoader_aging_young_old:reset(split)
  self.split_idxs[split] = 1
end


function DataLoader_aging_young_old:getBatch(split)
  local path_x = self.image_paths[string.format('%s_x', split)]
  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size - 1,
                           self.split_sizes[split])
  -- Load images out of the HDF5 file
  local images_x
  if split == 'train_young' or split == 'val' then 
  images_x = self.h5_file_young:read(path_x):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width}):float():div(255)
 
  elseif split == 'train_old' then
  images_x = self.h5_file_old:read(path_x):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width}):float():div(255) 
  end
  

  -- Advance counters, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.split_sizes[split] then
    self.split_idxs[split] = 1
  end
  -- Preprocess images
  images_pre_x = self.preprocess_fn(images_x) 
  return images_pre_x
end