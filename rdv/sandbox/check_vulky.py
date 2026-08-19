import vulky as vk
import torch

vk.create_device(debug=True)


buf = vk.structured_buffer(1024, element_description=dict(
    data=torch.int64,  # data ptr of the contiguous bound tensor
    grad_data=torch.int64,
    dim = int,  # dimension of the bound tensor
    shape=[8, int]  # dimensions of the bound tensor
), memory=vk.MemoryLocation.CPU)

buf2 = vk.structured_buffer(1024*1024, element_description=dict(
            name_id=int,  # name id of the key for the tensor.
            number_of_indices=int,  # number of indices used in this deferred parameter
            map_dim=int,  # dimension required by the map.
            indices=[4, int],  # indices set for the tensor
        ), memory=vk.MemoryLocation.CPU)

with buf2.map('inout') as b:
    b.number_of_indices[0] = 1

with buf.map('inout') as b:
    print(b.data)
    b.data = 1

with buf.map('inout') as b:
    print(b.data)
