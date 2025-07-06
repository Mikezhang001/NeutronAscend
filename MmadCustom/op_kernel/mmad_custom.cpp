 #include "kernel_operator.h"
 using namespace AscendC;

 constexpr int32_t QUEUE_DEPTH = 1;
 constexpr int32_t CUBE_SIZE = 16;
 
 class KernelMatmul
 {
 
 private:
     uint32_t m = 8192;
     uint32_t k = 68416;
     uint32_t n = 256;
     uint32_t buffer_num = 1;
     //理想的L1、L2的划分方式
     uint32_t L1_m = 0;
     uint32_t L1_k = 0;
     uint32_t L1_n = 0;
 
     uint32_t L2_m = 0;
     uint32_t L2_k = 0;
     uint32_t L2_n = 0;
 
     //实际的L1、L2的划分方式
     uint32_t L1_m_real = 0;
     uint32_t L1_k_real = 0;
     uint32_t L1_n_real = 0;
 
     uint32_t L2_m_real = 0;//L2_m_real = L1_m_real
     uint32_t L2_k_real = 0;
     uint32_t L2_n_real = 0;//L2_n_real = L1_n_real   
     
     //以L1_m，L1_n 划分结果矩阵区域
     uint32_t group_x = 0;
     uint32_t group_y = 0;
 
     uint32_t process_id = 0;
     uint32_t process_num = 0;
 
     uint32_t a_group_num = 0;
     uint32_t b_group_num = 0;
     uint32_t group_num = 0;
     
     bool is_new_group = true;
  
     TPipe pipe;

     TQue<TPosition::A1, QUEUE_DEPTH> inQueueA1;
     TQue<TPosition::A2, QUEUE_DEPTH> inQueueA2;
     TQue<TPosition::B1, QUEUE_DEPTH> inQueueB1;
     TQue<TPosition::B2, QUEUE_DEPTH> inQueueB2;
     // TQue<QuePosition::VECOUT, 1> csr_dense_VEC;
     // dst queue
     TQue<TPosition::CO1, QUEUE_DEPTH> outQueueCO1;
     TQue<TPosition::CO2, QUEUE_DEPTH> outQueueCO2;

     GlobalTensor<half> aGM, bGM;
     GlobalTensor<float> cGM;

 public:
     __aicore__ inline KernelMatmul()
     {
     }
     __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, int32_t M, int32_t N, int32_t K)
     {   
         KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
 
         this->m = M;
         this->n = N;
         this->k = K;
         this->buffer_num = 1;//double buffer没感到更快
        //分区策略: 传参
        //  this->L1_m = L1_m;
        //  this->L1_k = L1_k;
        //  this->L1_n = L1_n;
        //  this->L2_m = L2_m;
        //  this->L2_k = L2_k;
        //  this->L2_n = L2_n;

         this->InitData(M, N, K);// 计算L1、L2的划分方式
        //  AscendC::printf("L1_m %d, L1_k %d, L1_n %d, L2_m %d, L2_k %d, L2_n %d\n", 
        //         this->L1_m, this->L1_k, this->L1_n, this->L2_m, this->L2_k, this->L2_n);

         this->process_id = GetBlockIdx();
         this->process_num = GetBlockNum();
 
         // 以块为单位的行列总乘积，用来划分aicore的任务。
         this->a_group_num = this->CalcBlockNum(m, this->L1_m);
         this->b_group_num = this->CalcBlockNum(n, this->L1_n);
         this->group_num = this->a_group_num * this->b_group_num;
 
         aGM.SetGlobalBuffer((__gm__ half *)a);
         bGM.SetGlobalBuffer((__gm__ half *)b);
         cGM.SetGlobalBuffer((__gm__ float *)c);
 
         // L1 size=524288, L1_a和L1_各一半，double buffer, half占2个字节
         pipe.InitBuffer(inQueueA1, this->buffer_num, this->L1_m * this->L1_k * sizeof(half)); //(524288÷2)÷2÷2=256*256
         pipe.InitBuffer(inQueueB1, this->buffer_num, this->L1_k * this->L1_n * sizeof(half)); //(524288÷2)÷2÷2=256*256

         // L2 a size=65536,L2 b size=65536
         pipe.InitBuffer(inQueueA2, this->buffer_num, this->L2_m * this->L2_k * sizeof(half)); // 65536÷2÷2=256*64
         pipe.InitBuffer(inQueueB2, this->buffer_num, this->L2_k * this->L2_n * sizeof(half)); // 65536÷2÷2=256*64

         // l0_c_size=131072
         pipe.InitBuffer(outQueueCO1, this->buffer_num, this->L2_m * this->L2_n * sizeof(float)); //(131072)÷4÷2=256*64
        //  pipe.InitBuffer(outQueueCO2, this->buffer_num, this->L2_m * this->L2_n * sizeof(float));
     }
     __aicore__ inline void Process()
     {
         // 组分给aicore
         for (int group = this->process_id; group < this->group_num; group += this->process_num)
         {
             this->group_x = group / this->b_group_num;
             this->group_y = group % this->b_group_num;
             this->MatMul(group);

         }
     }
 
 private:
     __aicore__ inline void InitData(int M, int N, int K)
     {
         //分区策略: 对N进行切分, 多少组就切多少次,由L1_n体现分组
         int max_limit;
         int n_divide_16 = N / 16;
         int A1_size = 131072 / buffer_num; // l1_size大小为 524288字节，则A1可存放(524288÷2)÷2 个half数据
         int A2_size = 32768 / buffer_num; // l0_a_size大小为 65536字节，可存放half数据多少个
         int CO1_size = 32768 / buffer_num; // l0_c_size大小为 131072字节，可存放float数据多少个
         int group_count, block_sum, group_size; 

         if (n_divide_16 % 2 == 0) // 直接分两组
            group_count = 2;
         else
            group_count = 1;
            // group_count = n_divide_16;

         block_sum = GetBlockNum();//总共多少个aicore
         group_size = block_sum / group_count; // 每组aicore的数目

         this->L2_n = N / group_count;                                                                      // B做列切,(确保16的倍数)
         this->L1_n = this->L2_n; 

         this->L2_m = (M / group_size) > (CO1_size / this->L2_n) ? (CO1_size / this->L2_n) : (M / group_size); // 先定L2_m，受到CO1制约
         this->L2_m = this->L2_m / 16 * 16;                                                                  //(确保16的倍数)                                                                                        // 令window_H为16的倍数

         if(this->L2_m == 0) //修补M / group_size不够16导致为0
            this->L2_m = 16;                                                                                  
         this->L1_m = this->L2_m;
         
         max_limit = this->L2_n > this->L2_m ? this->L2_n : this->L2_m; // A2的L2_m和B2的L2_n都会对L2_k的选取产生影响
         this->L2_k = A2_size / max_limit;
         this->L2_k = this->L2_k / 16 * 16; //(确保16的倍数)

         this->L1_k = this->L2_k;            // 取相等绝不会溢出和犯错
        //  this->L1_k = A1_size / A2_size * this->L2_k;

     }
     __aicore__ inline void MatMul(int group)
     {   
         this->is_new_group = true;
 
         int k_split_max = this->CalcBlockNum(this->k, this->L1_k);
         for (int k_split_id = 0; k_split_id < k_split_max; k_split_id++)
         {
             this->Copy2L1inNZ(group, k_split_id);
             this->L2Operation();
         }
         
         this->Copyout(group);
     }
 
     __aicore__ inline void Copy2L1inNZ(int group, int k_split_id)
     {
         this->Copy2A1(group, k_split_id);
         this->Copy2B1(group, k_split_id);
     }
 
     __aicore__ inline void L2Operation()
     {   
         int L1_k_split_max = this->CalcBlockNum(this->L1_k_real, this->L2_k);
         
         for (int L1_k_split_id = 0; L1_k_split_id < L1_k_split_max; L1_k_split_id++)
         {
             this->NZCopy2A2ZZ(L1_k_split_id);
             this->NZCopy2B2ZN(L1_k_split_id);
             this->Compute();
         }
         LocalTensor<half> A1 = this->inQueueA1.DeQue<half>();
         this->inQueueA1.FreeTensor(A1);
         LocalTensor<half> B1 = this->inQueueB1.DeQue<half>();
         this->inQueueB1.FreeTensor(B1);
     }
 
     // L1_k需要是16的倍数
     __aicore__ inline void Copy2A1(int group, int k_split_id)
     {
         LocalTensor<half> A1 = this->inQueueA1.AllocTensor<half>();

         this->L1_m_real = this->m - (this->group_x * this->L1_m);
         this->L1_m_real = this->L1_m_real > this->L1_m ? this->L1_m : this->L1_m_real;
 
         this->L1_k_real = this->k - ( k_split_id * this->L1_k);
         this->L1_k_real = this->L1_k_real > this->L1_k ? this->L1_k : this->L1_k_real;
         //方式一：随路格式转换高级API进行数据搬运，但会有长度的限制,srcDValue∈[0, 65535]
         Nd2NzParams nd2nzA1Params;
         nd2nzA1Params.ndNum = 1;
         nd2nzA1Params.nValue = this->L1_m_real;
         nd2nzA1Params.dValue = this->L1_k_real;
         nd2nzA1Params.srcNdMatrixStride = 0;
         nd2nzA1Params.srcDValue = this->k;
         nd2nzA1Params.dstNzC0Stride = this->L1_m_real;
         nd2nzA1Params.dstNzNStride = 1;
         nd2nzA1Params.dstNzMatrixStride = 0;
         uint64_t global_offset = this->group_x * this->L1_m * this->k + k_split_id * this->L1_k;
         DataCopy(A1, aGM[global_offset], nd2nzA1Params);

         //#方式二：使用DataCopy进行数据搬运, srcStride∈[0, 65535 * 16]
        //  AscendC::DataCopyParams dataCopyParams;
        //  dataCopyParams.blockCount = this->L1_m_real;
        //  dataCopyParams.blockLen = 1;
        //  dataCopyParams.srcStride = (this->k - CUBE_SIZE) / CUBE_SIZE;
        //  dataCopyParams.dstStride = 0;
        //  uint64_t global_offset = (group / this->b_group_num) * this->L1_m * this->k + k_split_id * this->L1_k;
        //  for(int i = 0; i < this->CalcBlockNum(this->L1_k_real, CUBE_SIZE); i++){
        //      AscendC::DataCopy(A1[i * this->L1_m_real * CUBE_SIZE], aGM[global_offset], dataCopyParams);
        //      global_offset += CUBE_SIZE;
        //  }
 
         this->inQueueA1.EnQue<half>(A1);
     }
 
     __aicore__ inline void Copy2B1(int group, int k_split_id)
     {
         LocalTensor<half> B1 = this->inQueueB1.AllocTensor<half>();

         this->L1_k_real = this->k - ( k_split_id * this->L1_k);
         this->L1_k_real = this->L1_k_real > this->L1_k ? this->L1_k : this->L1_k_real;

         this->L1_n_real = this->n - (this->group_y * this->L1_n);
         this->L1_n_real = this->L1_n_real > this->L1_n ? this->L1_n : this->L1_n_real;
         //方式一：随路格式转换高级API进行数据搬运，但会有长度的限制,srcDValue∈[0, 65535]
         Nd2NzParams nd2nzB1Params;
         nd2nzB1Params.ndNum = 1;
         nd2nzB1Params.nValue = this->L1_k_real;
         nd2nzB1Params.dValue = this->L1_n_real;
         nd2nzB1Params.srcNdMatrixStride = 0;
         nd2nzB1Params.srcDValue = this->n;
         nd2nzB1Params.dstNzC0Stride = this->L1_k_real;
         nd2nzB1Params.dstNzNStride = 1;
         nd2nzB1Params.dstNzMatrixStride = 0;
         uint64_t global_offset = k_split_id * this->L1_k * this->n + this->group_y * this->L1_n;
         DataCopy(B1, bGM[global_offset], nd2nzB1Params);
        
         //#方式二：使用DataCopy进行数据搬运, srcStride∈[0, 65535 * 16]
        //  AscendC::DataCopyParams dataCopyParams;
        //  dataCopyParams.blockCount = this->L1_k_real;
        //  dataCopyParams.blockLen = 1;
        //  dataCopyParams.srcStride = (this->n - CUBE_SIZE) / CUBE_SIZE;
        //  dataCopyParams.dstStride = 0;
        //  uint64_t global_offset = k_split_id * this->L1_k * this->n + (group % this->b_group_num) * this->L1_n;
        //  for(int i = 0; i < this->CalcBlockNum(this->L1_n_real, CUBE_SIZE); i++){
        //      AscendC::DataCopy(B1[i * this->L1_k_real * CUBE_SIZE], bGM[global_offset], dataCopyParams);
        //      global_offset += CUBE_SIZE;
        //  }
 
         this->inQueueB1.EnQue<half>(B1);
     }
 
     __aicore__ inline void NZCopy2A2ZZ(int L1_k_split_id)
     {
         LocalTensor<half> A1 = this->inQueueA1.DeQue<half>();
         LocalTensor<half> A2 = this->inQueueA2.AllocTensor<half>();
 

         this->L2_k_real = this->L1_k_real - L1_k_split_id * this->L2_k;
         this->L2_k_real = this->L2_k_real > this->L2_k ? this->L2_k : this->L2_k_real;
         
         this->L2_m_real = this->L1_m_real;

         LoadData2DParams loadDataParams;
         //一行分形矩阵重复次数，就是一行有多少个分形矩阵
         loadDataParams.repeatTimes = this->CalcBlockNum(this->L2_k_real, CUBE_SIZE);
         //在nz格式中，前一个分行与后一个分形之间的地址偏移，单位为512B，即一个分形矩阵。
         loadDataParams.srcStride = this->CalcBlockNum(this->L1_m_real, CUBE_SIZE);
         loadDataParams.dstGap = 0;
         loadDataParams.ifTranspose = false;
 
         int src_start = L1_k_split_id * this->L2_k * this->L1_m_real;
         int row_cube_max = this->CalcBlockNum(this->L1_m_real, CUBE_SIZE);//L1_m_real = L2_m_real, L1_n_real = L2_n_real
         for (int i = 0; i < row_cube_max; i++)
         {   
             
             LoadData(A2[i * this->L2_k_real * CUBE_SIZE],
                     A1[src_start + i * CUBE_SIZE * CUBE_SIZE], loadDataParams);
         }
 
         this->inQueueA2.EnQue(A2);//L2缓存入队需要加类型模板,先入队A2，后入队A1，避免同步紊乱
         this->inQueueA1.EnQue(A1);
     }
 
     __aicore__ inline void NZCopy2B2ZN(int L1_k_split_id)
     {
         LocalTensor<half> B1 = this->inQueueB1.DeQue<half>();
         LocalTensor<half> B2 = this->inQueueB2.AllocTensor<half>();
 
         this->L2_k_real = this->L1_k_real - L1_k_split_id * this->L2_k;
         this->L2_k_real = this->L2_k_real > this->L2_k ? this->L2_k : this->L2_k_real;
         
         this->L2_n_real = this->L1_n_real;

         LoadData2DParams loadDataParams;
         loadDataParams.repeatTimes = this->CalcBlockNum(this->L2_n_real, CUBE_SIZE);
         loadDataParams.srcStride = this->CalcBlockNum(this->L1_k_real, CUBE_SIZE);
         loadDataParams.dstGap = 0;
         loadDataParams.ifTranspose = true;
 
         int src_start = L1_k_split_id * this->L2_k * CUBE_SIZE;
         int row_cube_max = this->CalcBlockNum(this->L2_k_real, CUBE_SIZE);
         for (int i = 0; i < row_cube_max; i++)
         {
             LoadData(B2[i * this->L2_n_real * CUBE_SIZE],
                     B1[src_start + i * CUBE_SIZE * CUBE_SIZE], loadDataParams);
         }
 
         this->inQueueB2.EnQue(B2);//避免同步紊乱
         this->inQueueB1.EnQue(B1);
     }
 
     __aicore__ inline void Compute()
     {
         LocalTensor<half> A2 = this->inQueueA2.DeQue<half>();
         LocalTensor<half> B2 = this->inQueueB2.DeQue<half>();
         LocalTensor<float> C1;
         if(this->is_new_group)
             C1 = this->outQueueCO1.AllocTensor<float>();
         else
             C1 = this->outQueueCO1.DeQue<float>();
         
         MmadParams mmadParams;
         mmadParams.m = this->L2_m_real;
         mmadParams.n = this->L2_n_real;
         mmadParams.k = this->L2_k_real;
         mmadParams.cmatrixInitVal = this->is_new_group;
         Mmad(C1, A2, B2, mmadParams);
         
         this->outQueueCO1.EnQue(C1);//先入队，后释放，避免同步紊乱
         this->inQueueA2.FreeTensor(A2);
         this->inQueueB2.FreeTensor(B2);
 
         this->is_new_group = false;
     }
 
     __aicore__ inline void Copyout(int group)
     {   
         // this->Copy2CO2(group);
         // this->Copy2GM(group);
         this->fixpipe2GM(group);
     }
 
     __aicore__ inline void fixpipe2GM(int group)
     {   
         LocalTensor<float> C1 = this->outQueueCO1.DeQue<float>();
         FixpipeParamsV220 fixpipeParams;//结果应该是L1_m_real * L1_n_real
         fixpipeParams.nSize = this->L1_n_real;
         fixpipeParams.mSize = this->L1_m_real;
         fixpipeParams.srcStride = this->L1_m_real;
         fixpipeParams.dstStride = this->n;
 
         fixpipeParams.ndNum = 1;
         fixpipeParams.srcNdStride = 0;
         fixpipeParams.dstNdStride = 0;
 
         uint64_t global_offset = this->group_x * this->L1_m * this->n + 
                             this->group_y * this->L1_n;
 
         Fixpipe(cGM[global_offset], C1, fixpipeParams);
         outQueueCO1.FreeTensor(C1);
     }
 
     __aicore__ inline void Copy2CO2(int group)
     {
         LocalTensor<float> C1 = this->outQueueCO1.DeQue<float>();
         LocalTensor<float> C2 = this->outQueueCO2.AllocTensor<float>();

         DataCopyParams dataCopyParams;
         dataCopyParams.blockCount = 1;
         dataCopyParams.blockLen = (this->L2_m / CUBE_SIZE) * (this->L2_n / CUBE_SIZE);

         DataCopyEnhancedParams enhancedParams;
         enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
         DataCopy(C2, C1, dataCopyParams, enhancedParams);

         // DataCopy(C2, C1, this->L2_m * this->L2_n);
 
         this->outQueueCO2.EnQue(C2);
         this->outQueueCO1.FreeTensor(C1);
     }
     
     __aicore__ inline void Copy2GM(int group)
     {   
         LocalTensor<float> C2 = this->outQueueCO2.DeQue<float>();
 
         DataCopyParams dataCopyParams;
         dataCopyParams.blockCount = this->L2_m;
         dataCopyParams.blockLen = 2;
         dataCopyParams.srcStride = 0;
         dataCopyParams.dstStride = (this->n - CUBE_SIZE) / 8;
 
         uint64_t global_offset = (group / this->b_group_num) * this->L1_m * this->n + 
                             (group % this->b_group_num) * this->L1_n;
         
         //每次取NZ的一列分形矩阵
         for(int i = 0; i < this->CalcBlockNum(this->L1_n, CUBE_SIZE); i++){
             DataCopy(cGM[global_offset], C2[i * this->L2_m * CUBE_SIZE], dataCopyParams);
             global_offset += CUBE_SIZE;
         }
 
         this->outQueueCO2.FreeTensor(C2);
     }
 
     __aicore__ inline int CalcBlockNum(int num, int unit_size)
     {
         return (num + unit_size - 1) / unit_size;
     }
 };

extern "C" __global__ __aicore__ void mmad_custom(GM_ADDR x, GM_ADDR y, GM_ADDR aicore_num, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelMatmul op;
    op.Init(x, y, z, tiling_data.m, tiling_data.n, tiling_data.k);
    op.Process();
}