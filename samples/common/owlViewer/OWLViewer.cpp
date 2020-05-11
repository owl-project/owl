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

#include "ViewerWidget.h"
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include "GL/glut.h"
#endif
#include "Camera.h"
#include "InspectMode.h"
#include "FlyMode.h"

namespace owl {
  namespace viewer {

    // ==================================================================
    // actual viewerwidget class
    // ==================================================================
    ViewerWidget::ViewerWidget(GlutWindow::SP window)
      : window(window),
        windowSize(window->getSize())
    {
    }

    void ViewerWidget::resize(const vec2i &newSize)
    {
      windowSize = newSize;
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
    void ViewerWidget::draw()
    {
      cudaGraphicsMapResources(1, &cuDisplayTexture);
      sample.copyGPUPixels(cuDisplayTexture);
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
    void ViewerWidget::updateCamera()
    {
      fullCamera.digestInto(simpleCamera);
      // if (isActive)
      cameraChanged();
    }

    void ViewerWidget::enableInspectMode(RotateMode rm,
                                         const box3f &validPoiRange,
                                         float minPoiDist,
                                         float maxPoiDist)
    {
      inspectModeManip
        = std::make_shared<InspectModeManip>(this,validPoiRange,minPoiDist,maxPoiDist,
                                             rm==POI? InspectModeManip::POI: InspectModeManip::Arcball);
      if (!cameraManip)
        cameraManip = inspectModeManip;
    }

    void ViewerWidget::enableInspectMode(const box3f &validPoiRange,
                                         float minPoiDist,
                                         float maxPoiDist)
    {
      enableInspectMode(POI,validPoiRange,minPoiDist,maxPoiDist);
    }

    void ViewerWidget::enableFlyMode()
    {
      flyModeManip
        = std::make_shared<FlyModeManip>(this);
      if (!cameraManip)
        cameraManip = flyModeManip;
    }

    /*! this gets called when the window determines that the mouse got
      _moved_ to the given position */
    void ViewerWidget::mouseMotion(const vec2i &newMousePosition)
    {
      if (lastMousePosition != vec2i(-1)) {
        if (leftButton.isPressed)   mouseDragLeft  (newMousePosition,newMousePosition-lastMousePosition);
        if (centerButton.isPressed) mouseDragCenter(newMousePosition,newMousePosition-lastMousePosition);
        if (rightButton.isPressed)  mouseDragRight (newMousePosition,newMousePosition-lastMousePosition);
      }
      lastMousePosition = newMousePosition;
    }

    void ViewerWidget::mouseDragLeft  (const vec2i &where, const vec2i &delta)
    {
      if (cameraManip) cameraManip->mouseDragLeft(where,delta);
    }

    /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
    void ViewerWidget::mouseDragCenter(const vec2i &where, const vec2i &delta)
    {
      if (cameraManip) cameraManip->mouseDragCenter(where,delta);
    }

    /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
    void ViewerWidget::mouseDragRight (const vec2i &where, const vec2i &delta)
    {
      if (cameraManip) cameraManip->mouseDragRight(where,delta);
    }

    /*! mouse button got either pressed or released at given location */
    void ViewerWidget::mouseButtonLeft  (const vec2i &where, bool pressed)
    {
      if (cameraManip) cameraManip->mouseButtonLeft(where,pressed);

      lastMousePosition = where;
    }

    /*! mouse button got either pressed or released at given location */
    void ViewerWidget::mouseButtonCenter(const vec2i &where, bool pressed)
    {
      if (cameraManip) cameraManip->mouseButtonCenter(where,pressed);

      lastMousePosition = where;
    }

    /*! mouse button got either pressed or released at given location */
    void ViewerWidget::mouseButtonRight (const vec2i &where, bool pressed)
    {
      if (cameraManip) cameraManip->mouseButtonRight(where,pressed);

      lastMousePosition = where;
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    void ViewerWidget::key(char key, const vec2i &where)
    {
      if (cameraManip) cameraManip->key(key,where);
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    void ViewerWidget::special(int key, const vec2i &where)
    {
      if (cameraManip) cameraManip->special(key,where);
    }

  } // ::owl::viewer
} // ::owl
