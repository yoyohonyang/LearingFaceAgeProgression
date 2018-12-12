require 'torch'
require 'nn'
require 'FaceAging.IdentityLoss'

local layer_utils = require 'FaceAging.layer_utils'
local crit, parent = torch.class('nn.PerceptualIdCriterion', 'nn.Criterion')

--[[
Input: args is a table with the following keys:
- net: A network calculating the loss.
- id_layers: An array of layer strings
- id_weights: A list of the same length as content_layers
- loss_type: Either "L2", or "SmoothL1", or "CrossEntropy"
--]]
function crit:__init(args)
  args.id_layers = args.id_layers or {}
  args.id_weights = args.id_weights or 1
  self.net = args.loss_id_net
  self.net:evaluate()
 
  self.id_loss_layers = {}

  local id_loss_layer
  
 -- Set up Id loss layers
  for i, layer_string in ipairs(args.id_layers) do   
    local weight = args.id_weights[i]	
	
	id_loss_layer = nn.IdentityLoss(weight, args.loss_type)
	print('We calculate the L2 LOSS or Smooth1 Loss!')
	
    layer_utils.insert_after(self.net, layer_string, id_loss_layer)
    table.insert(self.id_loss_layers, id_loss_layer)
	
  end
  print(self.net)
  layer_utils.trim_network(self.net)
  self.grad_net_output = torch.Tensor()
end

--[[
target: Tensor of shape (N, 3, H, W) giving pixels
--]]
function crit:setIdTarget(target)
	for i, id_loss_layer in ipairs(self.id_loss_layers) do		
		id_loss_layer:setMode('capture')	       
	end  	
    self.net:forward(target) 
end


function crit:setIdWeight(weight)
  for i, id_loss_layer in ipairs(self.id_loss_layers) do
    id_loss_layer.strength = weight
  end
end


--[[
Inputs:
- input: Tensor of shape (N, 3, H, W) giving pixels for generated images
- target: Table with the following keys:
  - ID_target: Tensor of shape (N, 3, H, W)
--]]
function crit:updateOutput(input, target)
  if target.id_target then
    self:setIdTarget(target.id_target)
  end
 
  -- Make sure to set all loss layers to loss mode before
  -- running the image forward.
  for i, id_loss_layer in ipairs(self.id_loss_layers) do
    id_loss_layer:setMode('loss')
  end
 
  local output = self.net:forward(input)

  -- Set up a tensor of zeros to pass as gradient to net in backward pass
  self.grad_net_output:resizeAs(output):zero()

  -- Go through and add up losses
  self.total_id_loss = 0
  self.id_losses = {}
  for i, id_loss_layer in ipairs(self.id_loss_layers) do
    self.total_id_loss = self.total_id_loss + id_loss_layer.loss
    table.insert(self.id_losses, id_loss_layer.loss)
  end
  self.output = self.total_id_loss
  return self.output
end


function crit:updateGradInput(input, target)
  self.gradInput = self.net:updateGradInput(input, self.grad_net_output)
  return self.gradInput
end


