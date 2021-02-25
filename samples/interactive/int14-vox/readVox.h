#pragma once

#include <string>
#include <vector>

#include "ogt_vox.h"

const ogt_vox_scene* loadVoxScene(const char *filename);

// Load multiple scenes and merge into one
const ogt_vox_scene* loadVoxScenes(const std::vector<std::string> &filenames);



