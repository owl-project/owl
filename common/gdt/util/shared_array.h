
  template<typename T>
  struct shared_array
  {
    shared_array()
      : N(0), ptr(nullptr)
    {}
    shared_array(size_t N)
      : sharedPtr(new T[N],[](T*t){delete[] t;}),
        N(N),
        ptr(sharedPtr.get())
    {}
    shared_array(T *t, size_t N)
      : N(N),
        ptr(t)
    { assert(t != nullptr); }
    shared_array(const shared_array &other) = default;
    shared_array(shared_array &&other) = default;
    ~shared_array() = default;

    inline size_t size() const { return N; }
    inline T     *get()  const { return ptr; }
  private:
    std::shared_ptr<T> sharedPtr;
    size_t             N;
    T                 *ptr;
  };

  
