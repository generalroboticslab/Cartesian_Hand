# CMakeLists.txt for EasyProfile library

# Add a library called 'EasyProfile' with all its source files
add_library(EasyProfile SHARED
  BasicTypes.h
  EasyObjectDictionary.cpp
  EasyObjectDictionary.h
  EasyProfile.cpp
  EasyProfile.h
  EasyProtocol.cpp
  EasyProtocol.h
  EasyQueue.h
)

# Set the include directory for EasyProfile
target_include_directories(EasyProfile PUBLIC .)