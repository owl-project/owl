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

    vec2i OWLViewer::getScreenSize()
    {
      int numModes = 0;
      const GLFWvidmode *modes
        = glfwGetVideoModes(glfwGetPrimaryMonitor(), &numModes);
      vec2i size(0,0);
      for (int i=0; i<numModes; i++) {
        size = max(size,vec2i(modes[i].width,modes[i].height));
      }
      return size;
    }
 
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
      // if (!cameraManipulator)
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
      // if (!cameraManipulator)
      cameraManipulator = flyModeManipulator;
    }

    /*! this gets called when the window determines that the mouse got
      _moved_ to the given position */
    void OWLViewer::mouseMotion(const vec2i &newMousePosition)
    {
      if (lastMousePosition != vec2i(-1)) {
        if (leftButton.isPressed)
          mouseDragLeft  (newMousePosition,newMousePosition-lastMousePosition);
        if (centerButton.isPressed)
          mouseDragCenter(newMousePosition,newMousePosition-lastMousePosition);
        if (rightButton.isPressed)
          mouseDragRight (newMousePosition,newMousePosition-lastMousePosition);
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
    void OWLViewer::special(int key, int mods, const vec2i &where)
    {
      if (cameraManipulator) cameraManipulator->special(key,where);
    }


    static void glfw_error_callback(int error, const char* description)
    {
      fprintf(stderr, "Error: %s\n", description);
    }
  


    OWLViewer::OWLViewer(const std::string &title,
                         const vec2i &initWindowSize)
    {
      glfwSetErrorCallback(glfw_error_callback);
      // glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
      
      if (!glfwInit())
        exit(EXIT_FAILURE);
      
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
      glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
      
      handle = glfwCreateWindow(initWindowSize.x, initWindowSize.y,
                                title.c_str(), NULL, NULL);
      if (!handle) {
        glfwTerminate();
        exit(EXIT_FAILURE);
      }
      
      glfwSetWindowUserPointer(handle, this);
      glfwMakeContextCurrent(handle);
      glfwSwapInterval( 1 );
    }


    /*! callback for a window resizing event */
    static void glfwindow_reshape_cb(GLFWwindow* window, int width, int height )
    {
      OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
      assert(gw);
      gw->resize(vec2i(width,height));
      // assert(OWLViewer::current);
      //   OWLViewer::current->resize(vec2i(width,height));
    }

    /*! callback for a key press */
    static void glfwindow_char_cb(GLFWwindow *window,
                                  unsigned int key) 
    {
      OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
      assert(gw);
      gw->key(key,gw->getMousePos());
    }

    /*! callback for a key press */
    static void glfwindow_key_cb(GLFWwindow *window,
                                 int key,
                                 int scancode,
                                 int action,
                                 int mods) 
    {
      OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
      assert(gw);
      if (action == GLFW_PRESS) {
        gw->special(key,mods,gw->getMousePos());
      }
    }

    /*! callback for _moving_ the mouse to a new position */
    static void glfwindow_mouseMotion_cb(GLFWwindow *window, double x, double y) 
    {
      OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
      assert(gw);
      gw->mouseMotion(vec2i((int)x, (int)y));
    }

    /*! callback for pressing _or_ releasing a mouse button*/
    static void glfwindow_mouseButton_cb(GLFWwindow *window, int button, int action, int mods) 
    {
      OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
      assert(gw);
      gw->mouseButton(button,action,mods);
    }
  
    void OWLViewer::mouseButton(int button, int action, int mods) 
    {
      const bool pressed = (action == GLFW_PRESS);
      switch(button) {
      case GLFW_MOUSE_BUTTON_LEFT:
        leftButton.isPressed = pressed;
        break;
      case GLFW_MOUSE_BUTTON_MIDDLE:
        centerButton.isPressed = pressed;
        break;
      case GLFW_MOUSE_BUTTON_RIGHT:
        rightButton.isPressed = pressed;
        break;
      }
      lastMousePos = getMousePos();
    }


    void OWLViewer::showAndRun()
    {
      int width, height;
      glfwGetFramebufferSize(handle, &width, &height);
      // PRINT(vec2i(width,height));
      resize(vec2i(width,height));

      // glfwSetWindowUserPointer(window, OWLViewer::current);
      glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
      glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
      glfwSetKeyCallback(handle, glfwindow_key_cb);
      glfwSetCharCallback(handle, glfwindow_char_cb);
      glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);
    
      while (!glfwWindowShouldClose(handle)) {
        static double lastCameraUpdate = -1.f;
        if (camera.lastModified != lastCameraUpdate) {
          cameraChanged();
          lastCameraUpdate = camera.lastModified;
        }
        render();
        draw();
        
        glfwSwapBuffers(handle);
        glfwPollEvents();
      }
    }
    
  } // ::owl::viewer
} // ::owl
