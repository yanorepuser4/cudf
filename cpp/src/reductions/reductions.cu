#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "utilities/type_dispatcher.hpp"

#include <cub/block/block_reduce.cuh>

#include <limits>
#include <type_traits>

#include "reduction_operators.cuh"

#define REDUCTION_BLOCK_SIZE 128

using namespace cudf::reduction;

namespace { // anonymous namespace


/*
Generic reduction implementation with support for validity mask
*/

template<typename T, typename F, typename Ld>
__global__
void gpu_reduction_op(const T *data, const gdf_valid_type *mask,
                      gdf_size_type size, T *results, F functor, T identity,
                      Ld loader)
{
    typedef cub::BlockReduce<T, REDUCTION_BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int step = blksz * gridsz;

    T agg = identity;

    for (int base=blkid * blksz; base<size; base+=step) {
        // Threadblock synchronous loop
        int i = base + tid;
        // load
        T loaded = identity;
        if (i < size && gdf_is_valid(mask, i))
            loaded = loader(data, i);

        // Block reduce
        T temp = BlockReduce(temp_storage).Reduce(loaded, functor);
        // Add current block
        agg = functor(agg, temp);
    }
    // First thread of each block stores the result.
    if (tid == 0)
        results[blkid] = agg;
}



template<typename T, typename F>
struct ReduceOp {
    static
    gdf_error launch(gdf_column *input, T identity, T *output,
                     gdf_size_type output_size) {
        // 1st round
        //    Partially reduce the input into *output_size* length.
        //    Each block computes one output in *output*.
        //    output_size == gridsize
        typedef typename F::Loader Ld1;
        F functor1;
        Ld1 loader1;
        launch_once((const T*)input->data, input->valid, input->size,
                    (T*)output, output_size, identity, functor1, loader1);
        CUDA_CHECK_LAST();

        return GDF_SUCCESS;
    }

    template <typename Functor, typename Loader>
    static
    void launch_once(const T *data, gdf_valid_type *valid, gdf_size_type size,
                     T *output, gdf_size_type output_size, T identity,
                     Functor functor, Loader loader) {

        // fake single step (single grid), the perf is bad.
        // Todo: using atomics (multi grid)
        output_size = 1;

        // find needed gridsize
        // use atmost REDUCTION_BLOCK_SIZE blocks
        int blocksize = REDUCTION_BLOCK_SIZE;
        int gridsize = (output_size < REDUCTION_BLOCK_SIZE?
                        output_size : REDUCTION_BLOCK_SIZE);

        // launch kernel
        gpu_reduction_op<<<gridsize, blocksize>>>(
            // inputs
            data, valid, size,
            // output
            output,
            // action
            functor,
            // identity
            identity,
            // loader
            loader
        );
    }

};

template <template <typename Ti> class Op>
struct ReduceDispatcher {

    template <typename T>
    gdf_error launch(gdf_column *col,
                         void *dev_result,
                         gdf_size_type dev_result_size)
    {
        GDF_REQUIRE(col->size > col->null_count, GDF_DATASET_EMPTY);
        T identity = Op<T>::identity();
        return ReduceOp<T, Op<T>>::launch(col, identity,
                                       static_cast<T*>(dev_result),
                                       dev_result_size);
    }

    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    gdf_error operator()(gdf_column *col,
                         void *dev_result,
                         gdf_size_type dev_result_size)
    {
        return launch<T>(col, dev_result, dev_result_size);
    }

    template <typename T, typename std::enable_if<
              !std::is_arithmetic<T>::value &&
              std::is_base_of<DeviceForNonArithmetic, Op<T>>::value
              >::type* = nullptr>
    gdf_error operator()(gdf_column *col,
                         void *dev_result,
                         gdf_size_type dev_result_size)
    {
        return launch<T>(col, dev_result, dev_result_size);
    }

    template <typename T, typename std::enable_if<
              !std::is_arithmetic<T>::value &&
              !std::is_base_of<DeviceForNonArithmetic, Op<T>>::value
              >::type* = nullptr>
    gdf_error operator()(gdf_column *col,
                         void *dev_result,
                         gdf_size_type dev_result_size) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};


}   // anonymous namespace

typedef enum {
  GDF_REDUCTION_SUM = 0,
  GDF_REDUCTION_MIN,
  GDF_REDUCTION_MAX,
  GDF_REDUCTION_PRODUCTION,
  GDF_REDUCTION_SUMOFSQUARES,
} gdf_reduction_op;


gdf_error gdf_reduction(gdf_column *col,
                  gdf_reduction_op op,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    switch(op){
    case GDF_REDUCTION_SUM:
        return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceSum>(),
                                     col, dev_result, dev_result_size);
    case GDF_REDUCTION_MIN:
        return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceMin>(),
                                     col, dev_result, dev_result_size);
    case GDF_REDUCTION_MAX:
        return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceMax>(),
                                     col, dev_result, dev_result_size);
    case GDF_REDUCTION_PRODUCTION:
        return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceProduct>(),
                                     col, dev_result, dev_result_size);
    case GDF_REDUCTION_SUMOFSQUARES:
        return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceSumOfSquares>(),
                                     col, dev_result, dev_result_size);
    default:
        { assert(false && "type_dispatcher: invalid gdf_type"); }
    }

    return GDF_INVALID_API_CALL;
}


gdf_error gdf_sum(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_SUM, dev_result, dev_result_size);
}

gdf_error gdf_product(gdf_column *col,
                      void *dev_result,
                      gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_PRODUCTION, dev_result, dev_result_size);
}

gdf_error gdf_sum_of_squares(gdf_column *col,
                             void *dev_result,
                             gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_SUMOFSQUARES, dev_result, dev_result_size);
}

gdf_error gdf_min(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_MIN, dev_result, dev_result_size);
}

gdf_error gdf_max(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_MAX, dev_result, dev_result_size);
}


unsigned int gdf_reduction_get_intermediate_output_size() {
    return REDUCTION_BLOCK_SIZE;
}
