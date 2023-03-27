c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) heng.cpp -o heng$(python3.7-config --extension-suffix)
cp heng$(python3.7-config --extension-suffix) /home/wjh/disk/pybind11test/
rm -f heng$(python3.7-config --extension-suffix)