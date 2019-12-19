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
#include "gdt/parallel/parallel_for.h"
// std
#include <vector>
#include <mutex>
#include <atomic>
#include <queue>

#define QBVH_DBG(a) /**/
#define QBVH_DBG_PRINT(a) QBVH_DBG(PRINT(a))
#define QBVH_DBG_PING QBVH_DBG(PING)

namespace gdt {
  namespace qbvh {

    struct BuildPrim {
      box3f bounds;
      int   primID;
    };
    inline std::ostream &operator<<(std::ostream &o, const BuildPrim &bp)
    { o << "{" << bp.primID << ":" << bp.bounds << "}"; return o; }

    /*! helper class that tracks both primitmive bounds _and_
      primitmive _centeroind_ bounds (as most builders reuquire */
    struct BothBounds {
      box3f prim, cent;

      void extend(BothBounds &other)
      { prim.extend(other.prim); cent.extend(other.cent); }
      
      void extend(const box3f &box)
      { prim.extend(box); cent.extend(box.center()); }
    };

    inline std::ostream &operator<<(std::ostream &o, const BothBounds &bb)
    { o << "{prim=" << bb.prim << ", cent=" << bb.cent << "}"; return o; }
    
    // static const int NUM_CHILDREN = 8;
      
    struct ChildRef {
      inline __both__ bool valid() const { return bits != 0; }

      inline __both__ void makeLeaf(uint32_t index)  { bits = index | (1UL<<31); }
      inline __both__ void makeInner(uint32_t index) { bits = index; }
      inline __both__ uint32_t getPrimIndex() const  { return bits & ~(1UL<<31); }
      inline __both__ uint32_t getChildIndex() const { return bits; }
      inline __both__ bool isLeaf() const { return (bits & (1UL<<31)); }
      uint32_t bits;
    };

    template<int NUM_CHILDREN>
    struct Node {
      vec3f origin;
      float width;
      struct { 
        uint8_t lo[NUM_CHILDREN];
        uint8_t hi[NUM_CHILDREN];
      } dim[3];
      ChildRef childRef[NUM_CHILDREN];

      void makeInner(int slot, const box3f &bounds, int childNodeID)
      {
        for (int d=0;d<3;d++) {
          const float lower_bounds = origin[d];
          const float upper_bounds = origin[d]+width; 
          assert(lower_bounds <= bounds.lower[d]);
          assert(upper_bounds >= bounds.upper[d]);
          dim[d].lo[slot] = max(0,min(255,int(256.f*(bounds.lower[d]-lower_bounds)/width)));
          dim[d].hi[slot] = max(0,min(255,int(256.f*(upper_bounds-bounds.upper[d])/width)));
        }
        childRef[slot].makeInner((unsigned)childNodeID);
      }
      void makeLeaf(int slot, const BuildPrim &bp)
      {
        for (int d=0;d<3;d++) {
          const float lower_bounds = origin[d];
          const float upper_bounds = origin[d]+width;
          assert(lower_bounds <= bp.bounds.lower[d]);
          assert(upper_bounds >= bp.bounds.upper[d]);
          dim[d].lo[slot] = max(0,min(255,int(256.f*(bp.bounds.lower[d]-lower_bounds)/width)));
          dim[d].hi[slot] = max(0,min(255,int(256.f*(upper_bounds-bp.bounds.upper[d])/width)));
        }
        childRef[slot].makeLeaf((unsigned)bp.primID);
      }

      inline __both__ box3f getBounds(int slot) const
      {
        const vec3i lo_i(dim[0].lo[slot],
                         dim[1].lo[slot],
                         dim[2].lo[slot]);
        const vec3i hi_i(dim[0].hi[slot],
                         dim[1].hi[slot],
                         dim[2].hi[slot]);
        const box3f box((origin)         + vec3f(lo_i) * (width/256.f),
                        (origin + width) - vec3f(hi_i) * (width/256.f));
        return box;
      }
      
      void initQuantization(const box3f &bounds)
      {
        origin = bounds.lower;
        width  = reduce_max(bounds.span())*1.0001f;
      }
      void  clearAllAfter(int maxValid)
      {
        for (int slot=maxValid;slot<NUM_CHILDREN;slot++) {
          // invalid leaf: no inner node can point to root node:
          childRef[slot].makeInner(0);
          for (int d=0;d<3;d++) {
            dim[d].lo[slot] = 255;
            dim[d].hi[slot] =   0;
          }
        }
      }

    };

    // template<typename PrimT>
    template<int NUM_CHILDREN>
    struct BVH {
      box3f              worldBounds;
      // std::vector<PrimT> prims;
      std::vector<Node<NUM_CHILDREN>>  nodes;
    };

    /*! a set of 'bin's that each track the centroid and primtiive
        boudns, as well as num prims, that project into this bin. used
        during binning. note that to better facilitate parallel
        binning we often have each thread/tbb job first bin "its"
        primtiives into its own local set of bins, then only at the
        end 'push' this local one back into tha 'master' bins. */
    template<int NUM_CHILDREN>
    struct Bins {

      /*! create a new set of bins over the given 'domain' (which
          should usually be the box over the centroids of the bins we
          are supposed to be binning */
      Bins(const box3f &domain)
        : domain(domain),
          scale(vec3f(numBins)*rcp(domain.upper-domain.lower + 1e-20f))
      {}

      /*! push one singel prim (with given prim bounds box) into this set of bins */
      inline void push(const box3f &b);
      
      /*! push an entire other set of bins into this set of bins */
      inline void push(const Bins &other);

      /*! the domain we're binning in; this should usually bound all
          incoming prims' centroids to start with, but just in case
          we'll always project back into the valid range, if only to
          handle nasty cases such as flat domains, numerically
          challenging cases, etc */
      const box3f domain;
      
      /*! precomputed 'numBins/domains.span(), to save on divisions */
      const vec3f scale;

      /*! num bins to use - could eventually be a template
          parameter */
      static const int numBins = 8;
      
      /*! @{ numBins x 3 bins : 3 dimensions (x,y,z), and numBins bins
          in each dimension */
      struct {
        struct {
          /*! bounding box of all prims that project into this bin (no
              need to track centbounds here, that'll be done during
              partition */
          box3f  bounds;
          /*! num prims in this bin */
          size_t count { 0 };
        } bin[numBins];
      } dim[3];
      /* @} */
    };

    /*! push one single prim (with given prim bounds box) into this set of bins */
    template<int NUM_CHILDREN>
    inline void Bins<NUM_CHILDREN>::push(const box3f &b)
    {
      const vec3f cent = b.center();
      const vec3i bin = max(vec3i(0),min(vec3i(numBins-1),vec3i((cent-domain.lower)*scale)));
      for (int d=0;d<3;d++) {
        dim[d].bin[bin[d]].bounds.extend(b);
        dim[d].bin[bin[d]].count++;
      }
    }
    
    /*! push an entire other set of bins into this set of bins */
    template<int NUM_CHILDREN>
    inline void Bins<NUM_CHILDREN>::push(const Bins<NUM_CHILDREN> &other)
    {
      for (int d=0;d<3;d++)
        for (int b=0;b<numBins;b++) {
          dim[d].bin[b].bounds.extend(other.dim[d].bin[b].bounds);
          dim[d].bin[b].count += other.dim[d].bin[b].count;
        }
    }

      
    template<int NUM_CHILDREN>
    inline void binIt(Bins<NUM_CHILDREN> &bins, const std::vector<BuildPrim> &bps,
               size_t begin, size_t end)
    {
      const size_t blockSize = 16*1024;
      if ((end-begin) > blockSize) {
        std::mutex mutex;
        parallel_for_blocked
          (begin,end,blockSize,[&](const size_t begin, const size_t end){
            Bins<NUM_CHILDREN> localBins(bins.domain);
            binIt(localBins,bps,begin,end);
            std::lock_guard<std::mutex> lock(mutex);
            bins.push(localBins);
          });
      } else {
        for (size_t i=begin;i<end;i++) {
          bins.push(bps[i].bounds);
        }
      }
    }

      
    template<int NUM_CHILDREN>
    struct SplitJob {
      SplitJob()
      {}
      SplitJob(const BothBounds &bounds,
               const size_t _begin,
               const size_t _end,
               std::vector<BuildPrim> *_src,
               std::vector<BuildPrim> *_dst)
        : bounds(bounds),
          begin(_begin),
          end(_end),
          src(_src),
          dst(_dst),
          priority(((_end-_begin)==1)?-1:area(bounds.prim))
      {
        initialized = true;
      }

      inline size_t size() { return end-begin; }

      bool initialized = false;
      BothBounds bounds;
      size_t begin=(size_t)-1, end=(size_t)-1;
      std::vector<BuildPrim> *src { nullptr };
      std::vector<BuildPrim> *dst { nullptr };
      float priority { -1.f };
    };
      
    template<int NUM_CHILDREN>
    struct SplitJobQueue {
      inline SplitJobQueue()
      {
        for (int i=0;i<(NUM_CHILDREN+1);i++)
          freeSlot[i] = &slot[i];
        numFree = NUM_CHILDREN+1;
      }
      inline size_t size() { return numActive; }
      inline SplitJob<NUM_CHILDREN> *alloc()
      {
        return freeSlot[--numFree];
      }
      inline void free(SplitJob<NUM_CHILDREN> *job)
      {
        freeSlot[numFree++] = job;
      }
      inline void insert(SplitJob<NUM_CHILDREN> *job)
      {
        activeSlot[numActive++] = job;
      }
      /*! get - but do NOT free - the given job */
      inline SplitJob<NUM_CHILDREN> *getActiveJob(int ID) { return activeSlot[ID]; }
      inline SplitJob<NUM_CHILDREN> *topAndPop() {
        assert(numActive > 0);
        int best = 0;
        for (int i=1;i<numActive;i++)
          if (activeSlot[i]->priority > activeSlot[best]->priority)
            best = i;
          
        SplitJob<NUM_CHILDREN> *bestJob = activeSlot[best];
        activeSlot[best] = activeSlot[--numActive];
        return bestJob;
      }
    private:
      SplitJob<NUM_CHILDREN> slot[NUM_CHILDREN+1];
      SplitJob<NUM_CHILDREN> *freeSlot[NUM_CHILDREN+1];
      SplitJob<NUM_CHILDREN> *activeSlot[NUM_CHILDREN+1];
      int numActive = 0;
      int numFree   = 0;
    };
      
    /*! job abstraction for splitting ONE subtree into MANY children */
    template<int NUM_CHILDREN>
    struct MultiNodeJob {
      MultiNodeJob(const size_t _allocedNodeID,
                   const BothBounds &bounds,
                   const size_t begin,
                   const size_t end,
                   std::vector<BuildPrim> *src,
                   std::vector<BuildPrim> *dst,
                   int depth
                   )
        : allocedNodeID(_allocedNodeID),
          bounds(bounds),
          begin(begin),
          end(end),
          src(src),
          dst(dst),
          depth(depth)
      {
      }

      size_t depth;
      size_t begin, end;
      size_t allocedNodeID;
      BothBounds bounds;
      std::vector<BuildPrim> *src;
      std::vector<BuildPrim> *dst;

      SplitJobQueue<NUM_CHILDREN> jobQueue;
      Node<NUM_CHILDREN> nodeToWrite;
        
      size_t size() { return end - begin; }

      void findBestSplit(SplitJob<NUM_CHILDREN> *in,
                         int                    &bestDim,
                         float                  &bestPos)
      {
        QBVH_DBG(std::cout << "---------------------------findSplit----------------------------" << std::endl);
        assert(in->initialized);
        bestDim = -1;
        assert(in->size() > 0);
        float bestCost = area(in->bounds.prim)*in->size();

        Bins<NUM_CHILDREN> bins(in->bounds.cent);
        assert(&in->src);
        binIt(bins,*in->src,in->begin,in->end);

        for (int d=0;d<3;d++) {
          box3f rBoundsArray[bins.numBins];
          box3f box;
          for (int i=bins.numBins-2;i>=0;--i) {
            box.extend(bins.dim[d].bin[i+1].bounds);
            rBoundsArray[i] = box;
          }
          size_t lCount = 0;
          size_t rCount = in->size();
          box3f lBounds;
          for (int i=0;i<bins.numBins-1;i++) {
            lCount += bins.dim[d].bin[i].count;
            lBounds.extend(bins.dim[d].bin[i].bounds);
            const size_t rCount = in->size()-lCount;
            const box3f  rBounds = rBoundsArray[i];
            if (!lCount || !rCount) continue;

            const float sah
              = lCount * area(lBounds)
              + rCount * area(rBounds);
            if (sah < bestCost) {
              bestCost = sah;
              bestDim  = d;
              bestPos
                = bins.domain.lower[d]
                + (bins.domain.upper[d]-bins.domain.lower[d])*(i+1)/float(bins.numBins);
            }
          }
        }
        QBVH_DBG(std::cout << "best split: " << bestPos << "@" << bestDim << std::endl;
                 std::cout << "   in " << in->bounds << std::endl);
      }
        
      bool tryToPartition(SplitJob<NUM_CHILDREN> *in,
                          SplitJob<NUM_CHILDREN> **childJob)
      {
        assert(in->size() > 0);
        assert(in->initialized);
        if (in->bounds.cent.lower == in->bounds.cent.upper) {
          // std::cout << "could not split (same centroid) : " 
          //           << begin << " " << end << " " << in->bounds << std::endl;
          return false;
        }

        int bestDim;
        float bestPos;

        findBestSplit(in,bestDim,bestPos);
        if (bestDim < 0)
          return false;

        BothBounds lBounds,rBounds;
        size_t mid
          = performPartition(bestDim,bestPos,*in->src,*in->dst,in->begin,in->end,
                             lBounds,rBounds);
        if (mid == in->begin || mid == in->end) {
          // std::cout << "could not split - no gain : " << begin << " " << end << " " << mid << std::endl;
          return false;
        }

        new(childJob[0])SplitJob<NUM_CHILDREN>(lBounds,in->begin,mid,in->dst,in->src);
        assert(childJob[0]->size() > 0);
               
        new(childJob[1])SplitJob<NUM_CHILDREN>(rBounds,mid,in->end,in->dst,in->src);
        assert(childJob[1]->size() > 0);

        return true;
      }
        
      /*! perform partition with given plane, and return final
        write pos where the two sides met */
      size_t performPartition(const int dim,
                              const float pos,
                              const std::vector<BuildPrim> &src,
                              std::vector<BuildPrim> &dst,
                              const size_t all_begin, const size_t all_end,
                              BothBounds &shared_lBounds,
                              BothBounds &shared_rBounds)
      {
        const size_t blockSize = 1024;
        std::atomic<size_t> shared_lPos(all_begin);
        std::atomic<size_t> shared_rPos(all_end);
          
        std::mutex mutex;

        // std::cout << " ====================== do partition ======================" << std::endl;
        parallel_for_blocked
          (all_begin,all_end,blockSize,[&](const size_t block_begin, const size_t block_end){
            // first, count *in out block*
            size_t Nl = 0;
            for (size_t i=block_begin;i<block_end;i++)
              if (src[i].bounds.center()[dim] < pos) Nl++;
            const size_t Nr = (block_end-block_begin)-Nl;
            // second, atomically 'allocate' in the output arrays
            size_t lPos = (shared_lPos+=Nl)-Nl;
            size_t rPos = (shared_rPos-=Nr);
            // finally - write ....
            BothBounds lBounds, rBounds;
            for (size_t i=block_begin;i<block_end;i++) {
              if (src[i].bounds.center()[dim] < pos) {
                lBounds.extend(src[i].bounds);
                dst[lPos++] = src[i];
              } else {
                rBounds.extend(src[i].bounds);
                dst[rPos++] = src[i];
              }
            }
            
            std::lock_guard<std::mutex> lock(mutex);
            shared_lBounds.extend(lBounds);
            shared_rBounds.extend(rBounds);
          });
        
        assert(shared_lPos == shared_rPos);
        QBVH_DBG(std::cout << "done partitioning " << (all_end-all_begin) << "@" << std::endl;
                 std::cout << "  -> l = " << (shared_lPos-all_begin) << ":" << shared_lBounds << std::endl;
                 std::cout << "  -> r = " << (all_end-shared_lPos) << ":" << shared_rBounds << std::endl;
                 );

        return shared_lPos;
      }

      void partitionInEqualHalves(SplitJob<NUM_CHILDREN> *in,
                                  SplitJob<NUM_CHILDREN> **childJob,
                                  int numFree)
      {
        QBVH_DBG_PING;
        QBVH_DBG(for (int i=in->begin;i<in->end;i++)
                   std::cout << " " << i << ": "<< in->src[i] << std::endl);
        int Nl = in->size() / 2;
        for (int i=in->begin;i<in->end;i++)
          (*in->dst)[i] = (*in->src)[i];
        new(childJob[0]) SplitJob<NUM_CHILDREN>(in->bounds,in->begin,in->begin+Nl,in->dst,in->src);
        new(childJob[1]) SplitJob<NUM_CHILDREN>(in->bounds,in->begin+Nl,in->end,in->dst,in->src);
      }

      void inlineOrPush(SplitJob<NUM_CHILDREN> *childJob)
      {
        // TODO: inline!
        jobQueue.insert(childJob);
      }
        
      /*! priority queue of build order: 'first' is the surface
        area of the subtree, if it still needs splitting, or -1,
        if it is to become a child node */
      void execute()
      {
        nodeToWrite.initQuantization(bounds.prim);
        // if (0 && size() < NUM_CHILDREN) {
        //   // THIS WILL NOT WORK - jobqueue is emptym, so eventually everything ets cleared out!
        //   /*! just force all chilren */
        //   for (int i=0;i<size();i++) {
        //     if (allocedNodeID==0)
        //       std::cout << "making leaf : " << i << std::endl;
        //     nodeToWrite.makeLeaf(i,(*src)[begin+i]);
        //   }
        // } else
        {
          SplitJob<NUM_CHILDREN> *job = jobQueue.alloc();
          new(job) SplitJob<NUM_CHILDREN>(bounds,begin,end,src,dst);
          jobQueue.insert(job);
          assert(job->initialized);
            
          while (jobQueue.size() < NUM_CHILDREN) {
            SplitJob<NUM_CHILDREN> *biggest_job = jobQueue.topAndPop();
            assert(biggest_job->initialized);
            if (biggest_job->priority < 0.f) {
              // won't be able to split this ... this is a LEAF!
              inlineOrPush(biggest_job);
              break;
            }
              
            SplitJob<NUM_CHILDREN> *childJob[2] = { jobQueue.alloc(),jobQueue.alloc() };
            assert(biggest_job->size() > 0);
            if (tryToPartition(biggest_job,childJob)) {
              assert(childJob[0]->size() > 0);
              assert(childJob[1]->size() > 0);
              inlineOrPush(childJob[0]);
              inlineOrPush(childJob[1]);
            } else {
              assert(biggest_job->size() >= 2);
              partitionInEqualHalves(biggest_job,childJob,NUM_CHILDREN-jobQueue.size());
              assert(childJob[0]->size() > 0);
              assert(childJob[1]->size() > 0);
              inlineOrPush(childJob[0]);
              inlineOrPush(childJob[1]);
            }
            jobQueue.free(biggest_job);
          }
        }
      }
    };
        

    template<int NUM_CHILDREN,
             // typename PrimT,
             // typename GetPrimLambda,
      typename GetBoundsLambda>
    struct Builder {
      Builder(BVH<NUM_CHILDREN>     &target,
              const size_t           numPrims,
              const GetBoundsLambda &getBounds);
        
        
      void buildInitialPrimBoundsAndWorldBounds(BothBounds &bounds,
                                                std::vector<BuildPrim> &buildPrims)
      {
        parallel_for_blocked(0,numPrims,1024,[&](const size_t begin, const size_t end){
            BothBounds blockBounds;
            std::vector<BuildPrim> blockPrims;
            for (size_t primID=begin;primID<end;primID++) {
              BuildPrim thisBP;
              thisBP.bounds = getBounds(primID);
              thisBP.primID = primID;
              if (thisBP.bounds.lower.x > thisBP.bounds.upper.x
                  ||
                  std::isnan(thisBP.bounds.lower.x))
                continue;
              blockPrims.push_back(thisBP);
              blockBounds.extend(thisBP.bounds);
            }
            std::lock_guard<std::mutex> lock(nodeArrayMutex);
            for (auto bp : blockPrims)
              buildPrims.push_back(bp);
            bounds.extend(blockBounds);
          });
      }

      size_t allocNode()
      {
        size_t thisNodeID = nextFreeNodeListSlot++;
        if (thisNodeID >= numReservedNodes) {
          std::lock_guard<std::mutex> lock(nodeArrayMutex);
          while(numReservedNodes <= thisNodeID) numReservedNodes += numReservedNodes;
          target.nodes.resize(numReservedNodes);
        }
        return thisNodeID;
        // return ;
      }
        
      void buildRec(MultiNodeJob<NUM_CHILDREN> &multiJob)
      {
        multiJob.execute();
        // serial_for(multiJob.jobQueue.size(),[&](int activeID) {
        parallel_for(multiJob.jobQueue.size(),[&](int activeID) {
            SplitJob<NUM_CHILDREN> *job = multiJob.jobQueue.getActiveJob(activeID);
            if (job->size() == 1) {
              multiJob.nodeToWrite.makeLeaf(activeID,(*job->src)[job->begin]);
            } else {
              MultiNodeJob<NUM_CHILDREN> childJob(allocNode(),job->bounds,
                                                  job->begin,job->end,
                                                  job->src,job->dst,
                                                  multiJob.depth+1);
              multiJob.nodeToWrite.makeInner(activeID,job->bounds.prim,childJob.allocedNodeID);
              buildRec(childJob);
            }                
          });
        multiJob.nodeToWrite.clearAllAfter(multiJob.jobQueue.size());
        {
          std::lock_guard<std::mutex> lock(nodeArrayMutex);
          target.nodes[multiJob.allocedNodeID] = multiJob.nodeToWrite;
        }
      }
    private:
      BVH<NUM_CHILDREN>     &target;
      const size_t           numPrims;
      const GetBoundsLambda &getBounds;
        
      std::mutex nodeArrayMutex;

      size_t           numReservedNodes     { 0 };
      std::atomic<int> nextFreeNodeListSlot { 0 };
      // std::atomic<int> nextFreePrimListSlot { 0 };
    };




      
    template<int NUM_CHILDREN, typename GetBoundsLambda>
    Builder<NUM_CHILDREN,GetBoundsLambda>::Builder(BVH<NUM_CHILDREN> &target,
                                                   const size_t           numPrims,
                                                   const GetBoundsLambda &getBounds)
        : target(target),
          numPrims(numPrims),
          getBounds(getBounds)
    {
      numReservedNodes = 1024;
      target.nodes.resize(numReservedNodes);
        
      std::vector<BuildPrim> buildPrim[2];
      buildPrim[0].reserve(numPrims);

      BothBounds rootBounds;
      buildInitialPrimBoundsAndWorldBounds(rootBounds,buildPrim[0]);
      target.worldBounds = rootBounds.prim;
      buildPrim[1].resize(buildPrim[0].size());
          
      MultiNodeJob<NUM_CHILDREN> rootJob(allocNode(),rootBounds,0,buildPrim[0].size(),
                                         &buildPrim[0],&buildPrim[1],0);
          
      buildRec(rootJob);
      target.nodes.resize(nextFreeNodeListSlot);
    }


      
    typedef BVH<4> BVH4;
    typedef BVH<8> BVH8;

    template<int NUM_CHILDREN, typename GetBoundsLambda>
    void build(BVH<NUM_CHILDREN>     &target,
               const size_t           numPrims,
               const GetBoundsLambda &getBounds)
    {
      Builder<NUM_CHILDREN,GetBoundsLambda>(target,numPrims,getBounds);
    }

  } // ::gdt::qbvh
} // ::gdt
