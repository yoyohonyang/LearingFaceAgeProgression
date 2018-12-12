require 'torch'
require 'hdf5'

local utils = require 'FaceAging.utils'
local preprocess = require 'FaceAging.preprocess'
local DataLoader_aging = torch.class('DataLoader_aging')


function DataLoader_aging:__init(opt)
  assert(opt.h5_file, 'Must provide h5_file')
  print(opt.h5_file)
  assert(opt.batch_size, 'Must provide batch size')
  self.preprocess_fn = preprocess[opt.preprocessing].preprocess
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  self.batch_size = opt.batch_size
  self.split_idxs = {
    train = 1,
    val = 1,
  } 
  self.image_paths = {
    train_x = '/train_x/images',
    val_x = '/val_x/images',
  }
  local train_size = self.h5_file:read(self.image_paths.train_x):dataspaceSize()

  self.split_sizes = {
    train = train_size[1],
    val = self.h5_file:read(self.image_paths.val_x):dataspaceSize()[1],
   
  }
  self.num_channels = train_size[2]
  self.image_height = train_size[3]
  self.image_width = train_size[4]

  self.num_minibatches = {}
  for k, v in pairs(self.split_sizes) do
    self.num_minibatches[k] = math.floor(v / self.batch_size)
  end

  
  if opt.max_train and opt.max_train > 0 then
    self.split_sizes.train = opt.max_train
  end
end


function DataLoader_aging:reset(split)
  self.split_idxs[split] = 1
end


function DataLoader_aging:getBatch(split)
  local path_x = self.image_paths[string.format('%s_x', split)]
  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size - 1,
                           self.split_sizes[split])
  -- Load images out of the HDF5 file
  local images_x = self.h5_file:read(path_x):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width}):float():div(255)
  
  -- Advance counters, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.split_sizes[split] then
    self.split_idxs[split] = 1
  end
  -- Preprocess images
  images_pre_x = self.preprocess_fn(images_x)
  return images_pre_x
end

