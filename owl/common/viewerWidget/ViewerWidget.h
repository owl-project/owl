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

#include "GlutWindow.h"
#include "Camera.h"

namespace owl {
  namespace viewer {

    // ------------------------------------------------------------------
    /*! a helper widget that opens and manages a viewer window,
      including some virtual mouse and frame buffer */
    struct OWL_VIEWER_INTERFACE ViewerWidget {
      GlutWindow::SP window;

      ViewerWidget(GlutWindow::SP window);


      /*! window notifies us that we got resized */
      virtual void resize(const vec2i &newSize) {
        window->resize(newSize);
        windowSize = newSize;
      }
      /*! gets called whenever the viewer needs us to re-render out widget */
      virtual void render() {}
      /*! gets called whenever glut has nothing else to do */
      virtual void idle() {};

      /*! draw framebuffer using OpenGL */
      virtual void drawGL() {}

      /*! gets called when the window gets shown for the very first time */
      virtual void activate()
      {
        isActive = true;
        updateCamera();
      }

      struct OWL_VIEWER_INTERFACE ButtonState {
        bool  isPressed        { false };
        vec2i posFirstPressed  { -1 };
        vec2i posLastSeen      { -1 };
        bool  shiftWhenPressed { false };
        bool  ctrlWhenPressed  { false };
      };

      ButtonState leftButton;
      ButtonState rightButton;
      ButtonState centerButton;
      vec2i       lastMousePosition { -1,-1 };

      /*! this gets called when the window determines that the mouse got
        _moved_ to the given position */
      virtual void mouseMotion(const vec2i &newMousePosition);

      /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
      virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta);
      /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
      virtual void mouseDragCenter(const vec2i &where, const vec2i &delta);
      /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
      virtual void mouseDragRight (const vec2i &where, const vec2i &delta);
      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonLeft  (const vec2i &where, bool pressed);
      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonCenter(const vec2i &where, bool pressed);
      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonRight (const vec2i &where, bool pressed);

      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void key(char key, const vec2i &/*where*/);
      /*! this gets called when the user presses a 'special' key on the keyboard (cursor keys) ... */
      virtual void special(int key, const vec2i &/*where*/);

      /*! re-draw the current frame. This function itself isn't
        virtual, but it calls the framebuffer's render(), which
        is */
      void draw();

      /*! idle callback - called whenever glut deems the window to be
        idle. if this function return true (ie, "yes, we _are_ idle)
        then we'll automatically do a usleep with the \see
        idle_usleep_delay value number of usecs; alternatively, the
        user can do "something" useful in this function (ie, update an
        animation and reutrn 0, in which case we directly post a
        redisplay without waiting */
      virtual bool idleFunction()
      { /* user to overwrite this */
        return true;
      }

      /*! set a new window aspect ratio for the camera, update the
        camera, and notify the app */
      void setAspect(const float aspect)
      {
        fullCamera.setAspect(aspect);
        updateCamera();
      }

      /*! set a new orientation for the camera, update the camera, and
        notify the app */
      void setCameraOrientation(/* camera origin    : */const vec3f &origin,
                                /* point of interest: */const vec3f &interest,
                                /* up-vector        : */const vec3f &up,
                                /* fovy, in degrees : */float fovyInDegrees)
      {
        //fullCamera.setOrientation(origin,interest,up,fovyInDegrees);
        fullCamera.setOrientation(origin,interest,up,fovyInDegrees,false);
        updateCamera();
      }


      void setCameraOptions(float fovy,
                            float focalDistance)

      {
        fullCamera.setFovy(fovy);
        fullCamera.setFocalDistance(focalDistance);
        updateCamera();
      }

      /*! this function gets called whenever any camera manipulator
        updates the camera. gets called AFTER all values have been updated */
      virtual void cameraChanged() {}

      /*! return currently active window size */
      vec2i getWindowSize() const { return windowSize; }

      const SimpleCamera &getCamera()     const
      {
        return simpleCamera;
      }

      std::shared_ptr<FullCameraManip> cameraManip;
      std::shared_ptr<FullCameraManip> inspectModeManip;
      std::shared_ptr<FullCameraManip> flyModeManip;

      void enableFlyMode();
      enum RotateMode { POI, Arcball };
      void enableInspectMode(RotateMode rm,
                             const box3f &validPoiRange=box3f(),
                             float minPoiDist=1e-3f,
                             float maxPoiDist=std::numeric_limits<float>::infinity());
      void enableInspectMode(const box3f &validPoiRange=box3f(),
                             float minPoiDist=1e-3f,
                             float maxPoiDist=std::numeric_limits<float>::infinity());
      void setWorldScale(const float worldScale)
      {
        fullCamera.motionSpeed = worldScale / sqrtf(3);
      }

      /*! re-computes the 'camera' from the 'cameracontrol', and notify
          app that the camera got changed */
      void updateCamera();

    private:
      friend struct GlutWindow;
      friend struct FullCameraManip;
      friend struct InspectModeManip;
      friend struct FlyModeManip;


    protected:
      vec2i  windowSize        { 0 };

      /*! a "preprocessed" camera that no longer tracks origin,
          diction, up-vector, fovy, focal distance etc,and instead
          only tracks the simplified 'focal plane and lens circle'
          parametrization */
      SimpleCamera simpleCamera;

      /*! the full camera state we are manipulating */
      FullCamera   fullCamera;

      /*! gets set to true when the window first gets shown */
      bool isActive { false };
    };

  } // ::owl::viewer
} // ::owl
