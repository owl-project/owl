#pragma once

#include <vector>
#include <vector_types.h>  // CUDA uchar4

struct VoxelModel {
    int dims[3];
    std::vector< uchar4 > voxels;
};

void readVox( const char* filename, std::vector< VoxelModel >& models, uchar4 palette[256] );

