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

#include "OWLViewer.h"
#include "Camera.h"
#include "InspectMode.h"
#include "FlyMode.h"

namespace owl {
  namespace viewer {


    float computeStableEpsilon(float f)
    {
      return abs(f) * float(1./(1<<21));
    }

    float computeStableEpsilon(const vec3f v)
    {
      return max(max(computeStableEpsilon(v.x),
                     computeStableEpsilon(v.y)),
                 computeStableEpsilon(v.z));
    }
    
    SimpleCamera::SimpleCamera(const Camera &camera)
    {
      auto &easy = *this;
      easy.lens.center = camera.position;
      easy.lens.radius = 0.f;
      easy.lens.du     = camera.frame.vx;
      easy.lens.dv     = camera.frame.vy;

      const float minFocalDistance
        = max(computeStableEpsilon(camera.position),
              computeStableEpsilon(camera.frame.vx));

      /*
        tan(fov/2) = (height/2) / dist
        -> height = 2*tan(fov/2)*dist
      */
      float screen_height
        = 2.f*tanf(camera.fovyInDegrees/2.f * (float)M_PI/180.f)
        * max(minFocalDistance,camera.focalDistance);
      easy.screen.vertical   = screen_height * camera.frame.vy;
      easy.screen.horizontal = screen_height * camera.aspect * camera.frame.vx;
      easy.screen.lower_left
        = //easy.lens.center
        /* NEGATIVE z axis! */
        - max(minFocalDistance,camera.focalDistance) * camera.frame.vz
        - 0.5f * easy.screen.vertical
        - 0.5f * easy.screen.horizontal;
      // easy.lastModified = getCurrentTime();
    }
    
    // ==================================================================
    // actual viewerwidget class
    // ==================================================================
    // OWLViewer::OWLViewer(GlutWindow::SP window)
    //   : window(window),
    //     windowSize(window->getSize())
    // {
    // }

    void OWLViewer::resize(const vec2i &newSize)
    {
      if (fbPointer)
        cudaFree(fbPointer);
      cudaMallocManaged(&fbPointer,newSize.x*newSize.y*sizeof(uint32_t));
      
      fbSize = newSize;
      if (fbTexture == 0) {
        glGenTextures(1, &fbTexture);
      } else {
        cudaGraphicsUnregisterResource(cuDisplayTexture);
      }
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, newSize.x, newSize.y, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, nullptr);
      
      // We need to re-register when resizing the texture
      cudaGraphicsGLRegisterImage(&cuDisplayTexture, fbTexture, GL_TEXTURE_2D, 0);
    }

    
    /*! re-draw the current frame. This function itself isn't
      virtual, but it calls the framebuffer's render(), which
      is */
    void OWLViewer::draw()
    {
      cudaGraphicsMapResources(1, &cuDisplayTexture);

      cudaArray_t array;
      cudaGraphicsSubResourceGetMappedArray(&array, cuDisplayTexture, 0, 0);
      {
        // sample.copyGPUPixels(cuDisplayTexture);
        cudaMemcpy2DToArray(array,
                            0,
                            0,
                            reinterpret_cast<const void *>(fbPointer),
                            fbSize.x * sizeof(uint32_t),
                            fbSize.x * sizeof(uint32_t),
                            fbSize.y,
                            cudaMemcpyDeviceToDevice);
      }
      cudaGraphicsUnmapResources(1, &cuDisplayTexture);
      
      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
    }

    /*! re-computes the 'camera' from the 'cameracontrol', and notify
      app that the camera got changed */
    void OWLViewer::updateCamera()
    {
      // camera.digestInto(simpleCamera);
      // if (isActive)
      camera.lastModified = getCurrentTime();
      cameraChanged();
    }

    void OWLViewer::enableInspectMode(RotateMode rm,
                                      const box3f &validPoiRange,
                                      float minPoiDist,
                                      float maxPoiDist)
    {
      inspectModeManipulator
        = std::make_shared<CameraInspectMode>
        (this,validPoiRange,minPoiDist,maxPoiDist,
         rm==POI? CameraInspectMode::POI: CameraInspectMode::Arcball);
      if (!cameraManipulator)
        cameraManipulator = inspectModeManipulator;
    }

    void OWLViewer::enableInspectMode(const box3f &validPoiRange,
                                      float minPoiDist,
                                      float maxPoiDist)
    {
      enableInspectMode(POI,validPoiRange,minPoiDist,maxPoiDist);
    }

    void OWLViewer::enableFlyMode()
    {
      flyModeManipulator
        = std::make_shared<CameraFlyMode>(this);
      if (!cameraManipulator)
        cameraManipulator = flyModeManipulator;
    }

    /*! this gets called when the window determines that the mouse got
      _moved_ to the given position */
    void OWLViewer::mouseMotion(const vec2i &newMousePosition)
    {
      if (lastMousePosition != vec2i(-1)) {
        if (leftButton.isPressed)   mouseDragLeft  (newMousePosition,newMousePosition-lastMousePosition);
        if (centerButton.isPressed) mouseDragCenter(newMousePosition,newMousePosition-lastMousePosition);
        if (rightButton.isPressed)  mouseDragRight (newMousePosition,newMousePosition-lastMousePosition);
      }
      lastMousePosition = newMousePosition;
    }

    void OWLViewer::mouseDragLeft  (const vec2i &where, const vec2i &delta)
    {
      if (cameraManipulator) cameraManipulator->mouseDragLeft(where,delta);
    }

    /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
    void OWLViewer::mouseDragCenter(const vec2i &where, const vec2i &delta)
    {
      if (cameraManipulator) cameraManipulator->mouseDragCenter(where,delta);
    }

    /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
    void OWLViewer::mouseDragRight (const vec2i &where, const vec2i &delta)
    {
      if (cameraManipulator) cameraManipulator->mouseDragRight(where,delta);
    }

    /*! mouse button got either pressed or released at given location */
    void OWLViewer::mouseButtonLeft  (const vec2i &where, bool pressed)
    {
      if (cameraManipulator) cameraManipulator->mouseButtonLeft(where,pressed);

      lastMousePosition = where;
    }

    /*! mouse button got either pressed or released at given location */
    void OWLViewer::mouseButtonCenter(const vec2i &where, bool pressed)
    {
      if (cameraManipulator) cameraManipulator->mouseButtonCenter(where,pressed);

      lastMousePosition = where;
    }

    /*! mouse button got either pressed or released at given location */
    void OWLViewer::mouseButtonRight (const vec2i &where, bool pressed)
    {
      if (cameraManipulator) cameraManipulator->mouseButtonRight(where,pressed);

      lastMousePosition = where;
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    void OWLViewer::key(char key, const vec2i &where)
    {
      if (cameraManipulator) cameraManipulator->key(key,where);
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    void OWLViewer::special(int key, const vec2i &where)
    {
      if (cameraManipulator) cameraManipulator->special(key,where);
    }




    OWLViewer::OWLViewer(const std::string &title)
                         // const vec3f &cameraInitFrom,
                         // const vec3f &cameraInitAt,
                         // const vec3f &cameraInitUp,
                         // const float worldScale)
    {
    }
    
    void OWLViewer::showAndRun()
    {
      PING;exit(0);
    }
    
  } // ::owl::viewer
} // ::owl
