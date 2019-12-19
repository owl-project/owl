// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "gdt/math/box.h"

namespace gdt {
  namespace pixelFormat {

    using RGB_F32 = gdt::vec3f;
    using RGB8    = gdt::vec3uc;

    /*! convert [0,1]-float to 8-bit fixed point represenation (ie, [0..255]) */
    inline __both__ uint8_t to_ufp8(float f)
    { 
      return uint8_t(255.f*f);
    }
    
    /*! convert vector of [0,1]-float to 8-bit fixed point represenation (ie, [0..255]) */
    inline __both__ vec3uc to_ufp8(const vec3f &v)
    {
      return vec3uc::make_from(v,[&](float f) { return to_ufp8(f); });
    }
  }
                       
  namespace image {
    
    /*! a 2D image, templated over a given pixel type */
    template<typename T>
    struct Image {
      __both__ Image(const vec2i &dims, T *_pixelPtr)
        : pixelPtr(_pixelPtr), //pixel(pixelPtr,area(dims)),
          dims(dims)
      {}
      
      /*! copy constructor - this only copies the *pointers*, not the
          actual array of pixels. Note we _need_ this constructor if
          we ever want to pass this to a cuda kernel, since the
          host-to-device passing _has_ to be copy-by-value, not
          copy-by-reference */
      Image(const Image &other)
        : dims(other.dims),
          pixelPtr(other.pixelPtr)
      {}

      Image(Image &&)      = default;

      /*! return the 2D image dimensstul */
      inline __both__ vec2i getDims() const { return dims; }

      /*! return a 2D-int bounding box that encloses all valid pixel
          IDs (ie, it does _NOT_ include the x=dims.x and y=dims.y
          coordinates!) */
      inline __both__ box2i getBounds() const { return box2i(getDims()-1); }

      /*! get the raw pointer to the pixel array */
      inline __both__ T *getData() const { return pixelPtr; }
      
      // inline __both__ T &get(size_t idx) const { return pixelPtr[idx]; };
      // inline __both__ T &get(const vec2i &idx) const { return get(idx.x+size_t(dims.x)*idx.y); }
      inline __both__ T &get(size_t idx) { return pixelPtr[idx]; };
      inline __both__ const T &get(size_t idx) const { return pixelPtr[idx]; };
      inline __both__ T &get(const vec2i &idx) { return get(idx.x+size_t(dims.x)*idx.y); }
      inline __both__ const T &get(const vec2i &idx) const { return get(idx.x+size_t(dims.x)*idx.y);}

      inline __both__ void set(const vec2i &idx, const T &value) { get(idx) = value; }
    private:
      //      shared_array<T> pixel;
      T    *pixelPtr = nullptr;
      vec2i dims;
    };

    using Image3f = Image<pixelFormat::RGB_F32>;
  }
  
} // ::gdt
