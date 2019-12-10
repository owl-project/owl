// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

#include "Device.h"

#define LOG(message)                                            \
  std::cout << "#owl.ll(" << context->owlDeviceID << "): "      \
  << message                                                    \
  << std::endl

#define LOG_OK(message)                                 \
  std::cout << GDT_TERMINAL_GREEN                       \
  << "#owl.ll(" << context->owlDeviceID << "): "        \
  << message << GDT_TERMINAL_DEFAULT << std::endl

#define CLOG(message)                                   \
  std::cout << "#owl.ll(" << owlDeviceID << "): "       \
  << message                                            \
  << std::endl

#define CLOG_OK(message)                                \
  std::cout << GDT_TERMINAL_GREEN                       \
  << "#owl.ll(" << owlDeviceID << "): "                 \
  << message << GDT_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {

    void Device::instanceGroupCreate(/*! the group we are defining */
                                     int groupID,
                                     /* list of children. list can be
                                        omitted by passing a nullptr, but if
                                        not null this must be a list of
                                        'childCount' valid group ID */
                                     int *childGroupIDs,
                                     /*! number of children in this group */
                                     int childCount)
    {
      assert("check for valid ID" && groupID >= 0);
      assert("check for valid ID" && groupID < groups.size());
      assert("check group ID is available" && groups[groupID] == nullptr);
        
      assert("check for valid combinations of child list" &&
             ((childGroupIDs == nullptr && childCount == 0) ||
              (childGroupIDs != nullptr && childCount >  0)));
        
      InstanceGroup *group
        = new InstanceGroup(childCount);
      assert("check 'new' was successful" && group != nullptr);
      groups[groupID] = group;
      
      // set children - todo: move to separate (api?) function(s)!?
      assert("currently have to specify all children at creation time" &&
             childCount > 0);
      assert("currently have to specify all children at creation time" &&
             childGroupIDs != nullptr);
      for (int childID=0;childID<childCount;childID++) {
        int childGroupID = childGroupIDs[childID];
        assert("check geom child child group ID is valid" && childGroupID >= 0);
        assert("check geom child child group ID is valid" && childGroupID <  groups.size());
        Group *childGroup = groups[childGroupID];
        assert("check referened child groups is valid" && childGroup != nullptr);
        childGroup->numTimesReferenced++;
        group->children[childID] = childGroup;
      }
    }

    void InstanceGroup::destroyAccel(Context *context) 
    {
      OWL_NOTIMPLEMENTED;
    }
    
    void InstanceGroup::buildAccel(Context *context) 
    {
      OWL_NOTIMPLEMENTED;
    }
    
  } // ::owl::ll
} //::owl
