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

#include "GlutWindow.h"

namespace owl {
  namespace viewer {
    
    struct ViewerWidget;

    /*! base abstraction for a camera that can generate rays. For this
      viewer, we assume we're dealine with a camera that has a
      rectangular viewing plane that's in focus, and a circular (and
      possible single-point) lens for depth of field. At some later
      point this should also capture time.{t0,t1} for motion blur, but
      let's leave this out for now. */
    struct OWL_VIEWER_INTERFACE SimpleCamera
    {
      struct {
        vec3f lower_left;
        vec3f horizontal;
        vec3f vertical;
      } screen;
      struct {
        vec3f center;
        vec3f du;
        vec3f dv;
        float radius { 0.f };
      } lens;
      /*! time (in seconds since system start) that camera was last
        modified */
      double lastModified { 0.f };
    };

    /*! the entire state for someting that can 'control' a camera -
        ie, that can rotate, move, focus, force-up, etc, a
        camera... for which it needs way more information than the
        simple camera.

        Note this uses a RIGHT HANDED camera as follows:
        - logical "up" is y axis
        - right is x axis
        - depth is _NEGATIVE_ z axis
    */
    struct OWL_VIEWER_INTERFACE FullCamera {
      FullCamera()
      {}

      /*! compute 'digest' of us as a "SimpleCamera", and set this
          object's fields */
      void digestInto(SimpleCamera &easy);

      vec3f getPOI() const
      {
        return position - poiDistance * frame.vz;
      }

      void setFovy(const float fovy);

      void setFocalDistance(float focalDistance);

      /*! set given aspect ratio */
      void setAspect(const float aspect);

      /*! re-compute all orientation related fields from given
          'user-style' camera parameters */
      void setOrientation(/* camera origin    : */const vec3f &origin,
                          /* point of interest: */const vec3f &interest,
                          /* up-vector        : */const vec3f &up,
                          /* fovy, in degrees : */float fovyInDegrees,
                          /* set focal dist?  : */bool  setFocalDistance=true);

      /*! tilt the frame around the z axis such that the y axis is "facing upwards" */
      void forceUpFrame();

      void setUpVector(const vec3f &up)
      { upVector = up; forceUpFrame(); }

      linear3f      frame         { one };
      vec3f         position      { 0,-1,0 };
      /*! distance to the 'point of interst' (poi); e.g., the point we
          will rotate around */
      float         poiDistance   { 1.f };
      float         focalDistance { 1.f };
      vec3f         upVector      { 0,1,0 };
      /* if set to true, any change to the frame will always use to
         upVector to 'force' the frame back upwards; if set to false,
         the upVector will be ignored */
      bool          forceUp       { true };

      /*! multiplier how fast the camera should move in world space
          for each unit of "user specifeid motion" (ie, pixel
          count). Initial value typically should depend on the world
          size, but can also be adjusted. This is actually something
          that should be more part of the manipulator widget(s), but
          since that same value is shared by multiple such widgets
          it's easiest to attach it to the camera here ...*/
      float         motionSpeed   { 1.f };
      float         aspect        { 1.f };
      float         fovyInDegrees { 60.f };
    };

    // ------------------------------------------------------------------
    /*! abstract base class that allows to manipulate a renderable
      camera */
    struct OWL_VIEWER_INTERFACE FullCameraManip {
      FullCameraManip(ViewerWidget *widget) : widget(widget) {}

      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void key(char key, const vec2i &where);

      /*! this gets called when the user presses a key on the keyboard ... */
      virtual void special(int key, const vec2i &where) { };

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta) {}

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragCenter(const vec2i &where, const vec2i &delta) {}

      /*! mouse got dragged with left button pressedn, by 'delta'
          pixels, at last position where */
      virtual void mouseDragRight (const vec2i &where, const vec2i &delta) {}

      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonLeft  (const vec2i &where, bool pressed) {}

      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonCenter(const vec2i &where, bool pressed) {}

      /*! mouse button got either pressed or released at given location */
      virtual void mouseButtonRight (const vec2i &where, bool pressed) {}

    protected:
      ViewerWidget *const widget;
    };

    // // ------------------------------------------------------------------
    // /*! game-sylte "WASD" camera manipulator. Mouse changes ray
    //   directoin, but its the WASD (and E,C) keys that move the
    //   positoin. Right mouse also moves forward/backward */
    // struct WasdFullCamera : public FullCamera {

    //   void move(const vec3f &camera_delta)
    //   {
    //     // const vec3f world_delta = xfmVector(frame,camera_delta);
    //     // camera.from += camera.motionSpeed * world_delta;
    //     // camera.at   += camera.motionSpeed * world_delta;
    //     // camera.dirty = true;
    //   }
    //   virtual void keyPress(char key, const vec2i &/*where*/)
    //   {
    //     switch(key) {
    //     case 'a': move(vec3f(-.1f,0,0)); return true;
    //     case 'd': move(vec3f(+.1f,0,0)); return true;

    //     case 'e': move(vec3f(0,0,-.1f)); return true;
    //     case 'q': move(vec3f(0,0,+.1f)); return true;

    //     case 's': move(vec3f(0,-.1f,0)); return true;
    //     case 'w': move(vec3f(0,+.1f,0)); return true;
    //     }
    //     return false;
    //   }

    // private:
    // };

  } // ::owl::viewer
} // ::owl

