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

// iw - include this first else glutwindow.h will include it without implementation
#define GL_LITE_IMPLEMENTATION 1
//#include "glew_lite.h"

#include "GlutWindow.h"
//#define GLEW_STATIC
//#define GL_GLEXT_PROTOTYPES 1
//#include <GL/glew.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include "glew_lite.h"
#include <GL/gl.h>
//#include <GL/glext.h>
#include <GL/glut.h>
#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif
#endif
#include "ViewerWidget.h"
//#include "../cuda.h"
//#include <cuda_runtime_api.h>
//#include <cuda_gl_interop.h>

namespace owl {
  namespace viewer {

    /*! gets set to true when glutInit() gets called - allows to make
        sure that this function got called before other glut calls get
        executed */
    bool glut_initialized = false;
    
    /*! actual implementation of a glut window. */
    struct GlutWindowImpl : public GlutWindow {
      GlutWindowImpl(int windowHandle,
                     GlutWindow::PixelFormat pixelFormat)
        : GlutWindow(windowHandle, pixelFormat),
          windowSize(1),
          widget(nullptr)
      {
      }
      
      virtual void setTitle(const std::string &title) override
      {
        glutSetWindow(glutWindowHandle);
        glutSetWindowTitle(title.c_str());
      }
      
      /*! viewer widget is done rendering, and asks the window to
        display these pixels.  These pixels _have_ to have the right
        format and size as indicated by pixelFormat and windowSize */
      virtual void drawPixels(void *pixels) override;

      /*! draw from a registered pbo */
      virtual void drawPBO(int pbo) override;
      

      /*! return current size (or more exactly, currectly alloc'ed
        size for internal buffers (optix in particular requires to
        make sure that its buffer size matches the PBO allocated
        buffer size, so needs to know how big that is even in the
        construtor, where 'resize' has never been called, yet */
      virtual vec2i getSize() const override
      {
        return windowSize;
      }
      

      virtual void idle() { assert(widget); widget->idle(); glutPostRedisplay(); };

      /*! key got pressed, at given location */
      virtual void key(unsigned char key, const vec2i &where)
      {
        if (widget) widget->key(key,where);
      }
      virtual void special(int key, const vec2i &where)
      {
        if (widget) widget->key(key,where);
      }
      
      /*! resize window */
      void resize(const vec2i &newSize) override
      {}
      
      /*! glut tells us window got resized */
      virtual void reshapeImpl(const vec2i &newSize)
      {
        windowSize = newSize;
        assert(widget);
        widget->resize(newSize);
      }

      /*! glut tells us window needs redisplay */
      void displayImpl()
      {
        if (widget)
          widget->render();
      }
    
      virtual void swapBuffers() override
      {
        glutSwapBuffers();
        glutPostRedisplay();
      }

      virtual void destroyWindow() override
      {
        glutDestroyWindow(glutWindowHandle);
      }
      
      /*! mouse was _moved_ */
      virtual void motion(const vec2i &pos)
      {
        if (widget) 
          widget->mouseMotion(pos);
      }

      /*! mouse gotted _clicked_ */
      virtual void mouse(int button, int state, const vec2i &pos)
      {
        switch(button) {
        case GLUT_LEFT_BUTTON: {
          widget->leftButton.isPressed = (state==GLUT_DOWN);
          widget->mouseButtonLeft(pos,widget->leftButton.isPressed);
        } break;
        case GLUT_RIGHT_BUTTON: {
          widget->rightButton.isPressed = (state==GLUT_DOWN);
          widget->mouseButtonRight(pos,widget->rightButton.isPressed);
        } break;
        case GLUT_MIDDLE_BUTTON: {
          widget->centerButton.isPressed = (state==GLUT_DOWN);
          widget->mouseButtonCenter(pos,widget->centerButton.isPressed);
        } break;
        };
      }

      
      /*! the viewer widget that this window is controlling */
      ViewerWidget *widget { nullptr };

      /*! latest set window size */
      vec2i windowSize { -1,-1 };

      GLuint glTextureID { 0 };
      
      static std::shared_ptr<GlutWindowImpl> current;
    };

    std::shared_ptr<GlutWindowImpl> GlutWindowImpl::current;

    /*! viewer widget is done rendering, and asks the window to
      display these pixels.  These pixels _have_ to have the right
      format and size as indicated by pixelFormat and windowSize */
    void GlutWindowImpl::drawPixels(void *imageData) 
    {
      glViewport(0, 0, windowSize.x, windowSize.y);
      
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
        
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

      
      static unsigned int gl_tex_id = 0;
      // Change these to GL_LINEAR for super- or sub-sampling
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      
      // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_TEXTURE_2D);
      switch(pixelFormat) {
      case GlutWindow::UINT8_RGBA:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowSize.x, windowSize.y,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, imageData);
        break;
      case GlutWindow::FLOAT3_RGB:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowSize.x, windowSize.y,
                     0, GL_RGB, GL_FLOAT, imageData);
        break;
      case GlutWindow::FLOAT4_RGBA:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize.x, windowSize.y,
                     0, GL_RGBA, GL_FLOAT, imageData);
        break;
      default:
        throw std::runtime_error("unknown pixel format");
      }

      glEnable(GL_TEXTURE_2D);

      glBegin(GL_QUADS);
      glTexCoord2f( 0.0f, 0.0f );
      glVertex2f  ( 0.0f, 0.0f );

      glTexCoord2f( 1.0f, 0.0f );
      glVertex2f  ( 1.0f, 0.0f );

      glTexCoord2f( 1.0f, 1.0f );
      glVertex2f  ( 1.0f, 1.0f );

      glTexCoord2f( 0.0f, 1.0f );
      glVertex2f  ( 0.0f, 1.0f );
      glEnd();

      glDisable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, 0);
    }
    


    /*! viewer widget is done rendering, and asks the window to
      display these pixels.  These pixels _have_ to have the right
      format and size as indicated by pixelFormat and windowSize */
    void GlutWindowImpl::drawPBO(int pbo) 
    {
      glViewport(0, 0, windowSize.x, windowSize.y);
      
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
        
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

      
      if(glTextureID <= 0) {
        glGenTextures( 1, &glTextureID );
        glBindTexture( GL_TEXTURE_2D, glTextureID );
        
        // Change these to GL_LINEAR for super- or sub-sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        
        // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      }

      glBindTexture( GL_TEXTURE_2D, glTextureID );
      glDisable(GL_DEPTH_TEST);
      
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
      glBindTexture( GL_TEXTURE_2D, glTextureID );
        
      switch(pixelFormat) {
      case GlutWindow::UINT8_RGBA:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowSize.x, windowSize.y,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        break;
      case GlutWindow::FLOAT3_RGB:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowSize.x, windowSize.y,
                     0, GL_RGB, GL_FLOAT, NULL);
        break;
      case GlutWindow::FLOAT4_RGBA:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize.x, windowSize.y,
                     0, GL_RGBA, GL_FLOAT, NULL);
        break;
      default:
        throw std::runtime_error("unknown pixel format");
      }
      glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );


      glEnable(GL_TEXTURE_2D);

      glBegin(GL_QUADS);
      glTexCoord2f( 0.0f, 0.0f );
      glVertex2f  ( 0.0f, 0.0f );

      glTexCoord2f( 1.0f, 0.0f );
      glVertex2f  ( 1.0f, 0.0f );

      glTexCoord2f( 1.0f, 1.0f );
      glVertex2f  ( 1.0f, 1.0f );

      glTexCoord2f( 0.0f, 1.0f );
      glVertex2f  ( 0.0f, 1.0f );
      glEnd();

      glDisable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, 0);
    }
    


    std::shared_ptr<GlutWindowImpl> getGlutWindow()
    {
      if (!GlutWindowImpl::current)
        throw std::runtime_error("trying to access current glut window, but no such window is active, yet");
      return GlutWindowImpl::current;
    }
    
    /*! initialize windowing system (typically glut) - has to be called exactly once per program */
    void GlutWindow::initGlut(int &ac, char **&av) 
    {
      if (glut_initialized)
        throw std::runtime_error("GlutWindow::initGlut has already been called!?");
      
      glutInit(&ac, av);
#ifndef __APPLE__
	  gl_lite_init();
#endif
      // glewInit();
      glut_initialized = true;
    }
    
    /*! callback for the glut 'display' event */
    void glut_display_cb()
    {
      getGlutWindow()->displayImpl();
    }
    
    /*! callback for the glut 'idle' event */
    void glut_idle_cb()
    {
      getGlutWindow()->idle();
      glutPostRedisplay();
    }
    
    /*! callback for the glut 'motion' event */
    void glut_motion_cb(int w, int h)
    {
      getGlutWindow()->motion(vec2i(w,h));
    }

    /*! callback for the glut 'mouse' event */
    void glut_mouse_cb(int button, int state, int w, int h)
    {
      getGlutWindow()->mouse(button,state,vec2i(w,h));
    }
    
    /*! callback for the glut 'reshape' event */
    void glut_reshape_cb(int w, int h)
    {
      
      getGlutWindow()->reshapeImpl(vec2i(w,h));
    }

    /*! callback for the glut 'keypress' event */
    void glut_key_cb(unsigned char key, int x, int y)
    {
      getGlutWindow()->key(key,vec2i(x,y));
    }

    /*! callback for the glut 'keypress' event */
    void glut_special_cb(int key, int x, int y)
    {
      getGlutWindow()->special(key,vec2i(x,y));
    }
    
    GlutWindow::SP GlutWindow::prepare(const vec2i &desiredSize,
                             GlutWindow::PixelFormat pixelFormat,
                             const std::string &initialTitle)
    {
      if (!glut_initialized)
        throw std::runtime_error("glut not yet initialized"
                                 " - did you forget to call GlutWindow::initGlut()?");

      glutInitDisplayMode(GLUT_RGB);
      glutInitWindowSize(desiredSize.x,desiredSize.y);
      glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
      return std::make_shared<GlutWindowImpl>(glutCreateWindow(initialTitle.c_str()),
                                              pixelFormat);
    }


    void GlutWindow::run(ViewerWidget &widget)
    {
      if (!glut_initialized)
        throw std::runtime_error("glut not yet initialized"
                                 " - did you forget to call GlutWindow::initGlut()?");
      if (GlutWindowImpl::current)
        throw std::runtime_error("GlutWindowImpl::current is non-null"
                                 " - is another window already running!?");

      GlutWindowImpl::current = std::dynamic_pointer_cast<GlutWindowImpl>(widget.window);
      if (!GlutWindowImpl::current)
        throw std::runtime_error("could not get glut window from widget !?");

      GlutWindowImpl::current->widget = &widget;
      
      glutDisplayFunc(glut_display_cb);
      glutKeyboardFunc(glut_key_cb);
      glutSpecialFunc(glut_special_cb);
      glutIdleFunc(glut_idle_cb);
      glutReshapeFunc(glut_reshape_cb);
      glutMotionFunc(glut_motion_cb);
      glutMouseFunc(glut_mouse_cb);
      
      glutShowWindow();


      widget.activate();

      
      glutMainLoop();
      
      GlutWindowImpl::current->widget = nullptr;
      GlutWindowImpl::current = nullptr;
    }

    void GlutWindow::quit(ViewerWidget &widget)
    {
      GlutWindowImpl::current = std::dynamic_pointer_cast<GlutWindowImpl>(widget.window);
      if (!GlutWindowImpl::current)
        throw std::runtime_error("could not get glut window from widget !?");

      GlutWindowImpl::current->destroyWindow();

#ifdef FREEGLUT
      glutLeaveMainLoop();
#endif

      GlutWindowImpl::current->widget = nullptr;
      GlutWindowImpl::current = nullptr;
    }
    

  } // ::owl::viewer
} // ::owl
