import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
import time


class MmadCustomTPNet(Cell):
    def __init__(self):
        super(MmadCustomTPNet, self).__init__()
        aclnn_ref_info = CustomRegOp("MmadCustomTP") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()
        self.op = ops.Custom("MmadCustomTP", lambda x_shape, y_shape: (x_shape[0], y_shape[1]), out_dtype=ms.float32, func_type="aot", bprop=None,
                             reg_info=aclnn_ref_info)
    def construct(self, x, y):
        res = self.op(x, y)
        return res

class MmadCustomTPAclnnNet(Cell):
    def __init__(self):
        super(MmadCustomTPAclnnNet, self).__init__()
        aclnn_ref_info = CustomRegOp("aclnnMmadCustomTP") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()
        self.op = ops.Custom("aclnnMmadCustomTP", lambda x_shape, y_shape: (x_shape[0], y_shape[1]), out_dtype=ms.float32, func_type="aot", bprop=None,
                             reg_info=aclnn_ref_info)
    def construct(self, x, y):
        res = self.op(x, y)
        return res

# context.set_context(mode=context.PYNATIVE_MODE,device_target="Ascend", device_id=6, jit_config={"jit_level": "O0"})
context.set_context(mode=context.GRAPH_MODE,device_target="Ascend", device_id=6, jit_config={"jit_level": "O0"})

x = Tensor(np.ones([16, 256]).astype(np.float16))
y = Tensor(np.ones([256, 256]).astype(np.float16))

#1.MmadCustomNet
# 通过lambda实现infer shape函数
# net = MmadCustomTPNet()

#2.MmadCustomAclnnNet
net = MmadCustomTPAclnnNet()

repeat_times = 3
for i in range(repeat_times):
      # print(f"第 {i} 次乘法开始")
      start_time1 = time.time()
      result = net(x, y)
      # timing_results.append(c)
      # ms.ops.Squeeze()(c).asnumpy()
      print(f"c的阻塞值为{result[0][0]}")
      # print(c.shape)
      start_time2 = time.time()
      print(f"第{i}轮的乘法耗时为{((start_time2 - start_time1) * 1000):.8f} ms")

print(result)