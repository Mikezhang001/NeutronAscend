import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
import time


class MmadCustomNet(Cell):
    def __init__(self):
        super(MmadCustomNet, self).__init__()
        aclnn_ref_info = CustomRegOp("MmadCustom") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()
        self.op = ops.Custom("MmadCustom", lambda x_shape, y_shape: (x_shape[0], y_shape[1]), out_dtype=ms.float32, func_type="aot", bprop=None,
                             reg_info=aclnn_ref_info)
    def construct(self, x, y):
        res = self.op(x, y)
        return res

class MmadCustomAclnnNet(Cell):
    def __init__(self):
        super(MmadCustomAclnnNet, self).__init__()
        aclnn_ref_info = CustomRegOp("aclnnMmadCustom") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()
        self.op = ops.Custom("aclnnMmadCustom", lambda x_shape, y_shape: (x_shape[0], y_shape[1]), out_dtype=ms.float32, func_type="aot", bprop=None,
                             reg_info=aclnn_ref_info)
    def construct(self, x, y):
        res = self.op(x, y)
        return res

# context.set_context(mode=context.PYNATIVE_MODE,device_target="Ascend", device_id=6, jit_config={"jit_level": "O0"})
context.set_context(mode=context.GRAPH_MODE,device_target="Ascend", device_id=6, jit_config={"jit_level": "O0"})

x = Tensor(np.ones([8192, 8192]).astype(np.float16))
y = Tensor(np.ones([8192, 256]).astype(np.float16))

#1.MmadCustomNet
# 通过lambda实现infer shape函数
# net = MmadCustomNet()

#2.MmadCustomAclnnNet
net = MmadCustomAclnnNet()

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