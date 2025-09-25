cmake_minimum_required(VERSION 3.15)
project(ev_cpp LANGUAGES CXX)

# If pybind11 is installed system-wide, scikit-build should find it automatically,
# but you can also explicitly find_package(pybind11) if needed.
find_package(pybind11 REQUIRED)

# This line tells scikit-build to create a Python extension module named "your_module_name"
# from the file `src/bindings.cpp`.
pybind11_add_module(ev_cpp src/bindings.cpp)

# If you want to link against other libraries or include additional files,
# you can do that here, e.g., target_link_libraries(your_module_name PRIVATE some_lib).

# By default, scikit-build will copy the compiled extension into your Python package folder
# if named appropriately. That’s typically handled by the wheel-building logic automatically.