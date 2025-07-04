
#include "mmad_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  MmadCustomTilingData tiling;
  uint32_t m = static_cast<uint32_t>((context->GetInputTensor(0)->GetOriginShape())[0]);//获取输入张量X[0]
  uint32_t k = static_cast<uint32_t>((context->GetInputTensor(0)->GetOriginShape())[1]);//获取输入张量X[1]
  uint32_t n = static_cast<uint32_t>((context->GetInputTensor(1)->GetOriginShape())[1]);//获取输入张量Y[1]
  
  tiling.set_m(m);
  tiling.set_k(k);
  tiling.set_n(n);
  context->SetBlockDim(20);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}
}


// namespace ge {
// static ge::graphStatus InferShape(gert::InferShapeContext* context)
// {
//     // const gert::Shape* x1_shape = context->GetInputShape(0);
//     // gert::Shape* y_shape = context->GetOutputShape(0);
//     // *y_shape = *x1_shape;
//     return GRAPH_SUCCESS;
// }
// }

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* y_shape = context->GetInputShape(1);


    gert::Shape* z_shape = context->GetOutputShape(0);
    z_shape->SetDim(0, x_shape->GetDim(0)); // z 的第 0 维等于 x 的第 0 维
    z_shape->SetDim(1, y_shape->GetDim(1)); // z 的第 1 维等于 y 的第 1 维
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // const auto inputDataType = context->GetInputDataType(0);
    // context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(0, ge::DataType::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MmadCustom : public OpDef {
public:
    explicit MmadCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // this->SetInferShape(ge::InferShape);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        
        
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MmadCustom);
}
