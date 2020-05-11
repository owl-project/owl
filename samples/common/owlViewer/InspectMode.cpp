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

#include "InspectMode.h"
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include "ViewerWidget.h"

namespace owl {
  namespace viewer {

    const float kbd_rotate_degrees = 10.f;
    const float degrees_per_drag_fraction = 150;
    const float pixels_per_move = 10.f;

    InspectModeManip::InspectModeManip(ViewerWidget *widget,
                                       const box3f &validPoiBounds,
                                       const float minDistance,
                                       const float maxDistance,
                                       const RotateMode rotateMode)
      : FullCameraManip(widget),
        validPoiBounds(validPoiBounds),
        minDistance(minDistance),
        maxDistance(maxDistance),
        rotateMode(rotateMode)
    {
    }

    void InspectModeManip::setRotateMode(RotateMode rm)
    {
      rotateMode = rm;
    }

    // ##################################################################
    // actual motion functions that do the actual work
    // ##################################################################

    void InspectModeManip::move(const float step)
    {
      FullCamera &fc = widget->fullCamera;
      
      const vec3f poi  = fc.getPOI();
      fc.poiDistance   = min(maxDistance,max(minDistance,fc.poiDistance-step*fc.motionSpeed));
      //      fc.focalDistance = fc.poiDistance;
      fc.position = poi + fc.poiDistance * fc.frame.vz;
      widget->updateCamera();
    }

    void InspectModeManip::movePOI(const vec2f step)
    {
      FullCamera &fc = widget->fullCamera;
      
      const vec3f old_poi  = fc.getPOI();
      vec3f new_poi = old_poi - step.x * fc.motionSpeed * fc.frame.vx + step.y * fc.motionSpeed * fc.frame.vy;
      if (!validPoiBounds.empty())
        new_poi = max(min(new_poi,validPoiBounds.upper),validPoiBounds.lower);
      fc.position += (new_poi - old_poi);
      widget->updateCamera();
    }

    vec3f InspectModeManip::arcballProject(const vec2i &where)
    {
      int width  = widget->getWindowSize().x;
      int height = widget->getWindowSize().y;

      vec3f v(0.f);

      v.x =  (where.x - .5f * width ) / (.5f * width );
      v.y = -(where.y - .5f * height) / (.5f * height);

      float d = v.x*v.x+v.y*v.y;

      if (d > 1.f) {
        float length = sqrtf(d);

        v.x /= length;
        v.y /= length;
      } else
        v.z = sqrtf(1.f-d);

      return v;
    }

    void InspectModeManip::rotate(const float deg_u,
                                  const float deg_v)
    {
      float rad_u = -(float)M_PI/180.f*deg_u;
      float rad_v = -(float)M_PI/180.f*deg_v;

      assert(widget);
      FullCamera &fc = widget->fullCamera;
      
      const vec3f poi  = fc.getPOI();
      fc.frame
        = linear3f::rotate(fc.frame.vy,rad_u)
        * linear3f::rotate(fc.frame.vx,rad_v)
        * fc.frame;

      if (fc.forceUp) fc.forceUpFrame();

      fc.position = poi + fc.poiDistance * fc.frame.vz;
      
      widget->updateCamera();
    }

    // ##################################################################
    // MOUSE interaction
    // ##################################################################

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    void InspectModeManip::mouseDragLeft(const vec2i &where, const vec2i &delta)
    {
      if (rotateMode == Arcball) {
        vec3f curr_pos = arcballProject(where+delta);
        vec3f from = normalize(arcball.down_pos);
        vec3f to = normalize(curr_pos);
        arcball.rotation = Quaternion3f(dot(from,to),cross(from,to)) * arcball.down_rotation;

        LinearSpace3f rotation_matrix(conj(arcball.rotation));

        assert(widget);
        FullCamera &fc = widget->fullCamera;

        vec3f eye(0.f,0.f,length(fc.getPOI()-fc.position));
        eye = rotation_matrix * eye;
        eye += fc.getPOI();

        vec3f up = rotation_matrix.vy;

        fc.setOrientation(eye,fc.getPOI(),up,fc.fovyInDegrees,fc.focalDistance);

        if (fc.forceUp) fc.forceUpFrame();

        widget->updateCamera();
      } else {
        const vec2f fraction = vec2f(delta) / vec2f(widget->windowSize);
        rotate(fraction.x * degrees_per_drag_fraction,
               fraction.y * degrees_per_drag_fraction);
      }
    }

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    void InspectModeManip::mouseDragCenter(const vec2i &where, const vec2i &delta)
    {
      const vec2f fraction = vec2f(delta) / vec2f(widget->windowSize);
      movePOI(fraction*pixels_per_move);
    }

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    void InspectModeManip::mouseDragRight(const vec2i &where, const vec2i &delta)
    {
      const vec2f fraction = vec2f(delta) / vec2f(widget->windowSize);
      move(fraction.y*pixels_per_move);
    }

    void InspectModeManip::mouseButtonLeft(const vec2i &where, bool pressed)
    {
      if (rotateMode == Arcball && pressed) {
        if (abs(arcball.rotation) < 1e-8f) {
          FullCamera &fc = widget->fullCamera;
          arcball.rotation = LinearSpace3f::rotation(fc.frame);
        }
        arcball.down_pos = arcballProject(where);
        arcball.down_rotation = arcball.rotation;
      }
    }

    // ##################################################################
    // KEYBOARD interaction
    // ##################################################################

    void InspectModeManip::kbd_up()
    {
      rotate(0,+kbd_rotate_degrees);
    }
    
    void InspectModeManip::kbd_down()
    {
      rotate(0,-kbd_rotate_degrees);
    }

    /*! keyboard left/right: note this works _exactly_ the other way
        around than the camera does: moving camera to the right
        'drags' the model to the right (ie, the camer to the left),
        but _typing_ right 'moves' the viewer/camera, so rotates
        camera to the right). This _reads_ counter-intuitive, but
        feels more natural, so is intentional */
    void InspectModeManip::kbd_right()
    {
      rotate(-kbd_rotate_degrees,0);
    }
    
    /*! keyboard left/right: note this works _exactly_ the other way
        around than the camera does: moving camera to the right
        'drags' the model to the right (ie, the camer to the left),
        but _typing_ right 'moves' the viewer/camera, so rotates
        camera to the right). This _reads_ counter-intuitive, but
        feels more natural, so is intentional */
    void InspectModeManip::kbd_left()
    {
      rotate(+kbd_rotate_degrees,0);
    }
    
    void InspectModeManip::kbd_forward()
    {
      move(+1.f);
    }
    
    void InspectModeManip::kbd_back()
    {
      move(-1.f);
      // FullCamera &fc = widget->fullCamera;
      // float step = 1.f;
      
      // const vec3f poi  = fc.position - fc.poiDistance * fc.frame.vz;
      // fc.poiDistance   = max(maxDistance,fc.poiDistance+step);
      // fc.focalDistance = fc.poiDistance;
      // fc.position = poi + fc.poiDistance * fc.frame.vz;
      // widget->updateCamera();
    }
    
    
    /*! this gets called when the user presses a key on the keyboard ... */
    void InspectModeManip::key(char key, const vec2i &where) 
    {
      FullCamera &fc = widget->fullCamera;
      
      switch(key) {
      case 'w':
        kbd_up();
        break;
      case 's':
        kbd_down();
        break;
      case 'd':
        kbd_right();
        break;
      case 'a':
        kbd_left();
        break;
      case 'e':
        kbd_forward();
        break;
      case 'c':
        kbd_back();
        break;
      default:
        FullCameraManip::key(key,where);
      }
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    void InspectModeManip::special(int key, const vec2i &where) 
    {
      switch (key) {
      case GLUT_KEY_UP:
        kbd_up();
        break;
      case GLUT_KEY_DOWN:
        kbd_down();
        break;
      case GLUT_KEY_RIGHT:
        kbd_right();
        break;
      case GLUT_KEY_LEFT:
        kbd_left();
        break;
      case GLUT_KEY_PAGE_UP:
        kbd_forward();
        break;
      case GLUT_KEY_PAGE_DOWN:
        kbd_back();
        break;
      }
    }

  } // ::owl::viewer
} // ::owl
