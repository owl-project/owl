// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

#if defined(_MSC_VER)
#  define OWL_VIEWER_DLL_EXPORT __declspec(dllexport)
#  define OWL_VIEWER_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define OWL_VIEWER_DLL_EXPORT __attribute__((visibility("default")))
#  define OWL_VIEWER_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define OWL_VIEWER_DLL_EXPORT
#  define OWL_VIEWER_DLL_IMPORT
#endif

#if defined(owl_viewer_DLL_INTERFACE)
#  ifdef owl_viewer_EXPORTS
#    define OWL_VIEWER_INTERFACE OWL_VIEWER_DLL_EXPORT
#  else
#    define OWL_VIEWER_INTERFACE OWL_VIEWER_DLL_IMPORT
#  endif
#else
#  define OWL_VIEWER_INTERFACE /*static lib*/
#endif
#include "owl/common/math/box.h"
#include "owl/common/math/LinearSpace.h"

#include <vector>
#include <memory>
#ifdef _GNUC_
    #include <unistd.h>
#endif

namespace owl {
  namespace viewer {
    using namespace owl::common;
    
    struct ViewerWidget;
    
    /*! abstraction for the glut window controlled by this library;
      the actual implementation will remain hidden */
    struct OWL_VIEWER_INTERFACE GlutWindow {
      /*! short-hand Class::SP instead of std::shared_ptr<Class>, for
        readability */
      typedef std::shared_ptr<GlutWindow> SP;
      
      typedef enum {
        UINT8_RGBA,
        FLOAT_RGB,
        FLOAT3_RGB=FLOAT_RGB,
        FLOAT_RGBA,
        FLOAT4_RGBA=FLOAT_RGBA
      } PixelFormat;
      
      GlutWindow(int glutWindowHandle, PixelFormat pixelFormat)
        : glutWindowHandle(glutWindowHandle),
          pixelFormat(pixelFormat)
      {}
      
      /*! should be called once, before anything else gets done with
        any window creation etc */
      static void initGlut(int &ac, char **&av);
      
      /*! viewer widget is done rendering, and asks the window to
        display these pixels.  These pixels _have_ to have the right
        format and size as indicated by pixelFormat and windowSize */
      virtual void drawPixels(void *pixels) = 0;
      
      /*! draw from a registered pbo */
      virtual void drawPBO(int pbo) = 0;
      
      static void run(ViewerWidget &widget);
      static GlutWindow::SP prepare(const vec2i &desiredSize,
                                    PixelFormat pixelType,
                                    const std::string &initialTitle);
      static void quit(ViewerWidget &widget);

      virtual void swapBuffers() = 0;
      virtual void destroyWindow() = 0;
      virtual void setTitle(const std::string &title) = 0;

      /*! resize the PBO, texture, etc */
      virtual void resize(const vec2i &newSize) = 0;

      /*! return current size (or more exactly, currectly alloc'ed
          size for internal buffers (optix in particular requires to
          make sure that its buffer size matches the PBO allocated
          buffer size, so needs to know how big that is even in the
          construtor, where 'resize' has never been called, yet */
      virtual vec2i getSize() const = 0;
      
      ViewerWidget *widget { nullptr };
      /*! window handle, in case we need to create a GLUI window, for
        example */
      int               glutWindowHandle { -1 };
      const PixelFormat pixelFormat;
    };
    
  } // ::owl::viewer
} // ::owl

